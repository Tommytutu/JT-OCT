"""D3-contracted JT min-sum for D4/D5, with certified message intervals.

Each terminal factor is the exact cost of a depth-three subtree conditional on
its ancestor separator. Upper split costs are owned once by their upper node.
This is a coarsened junction-tree representation, not the original path chain.
"""
from collections import OrderedDict
from dataclasses import asdict,dataclass,replace
from itertools import islice
import math
import sys
import time

import numpy as np

from .cg import greedy_feasible
from .d3_optimized import D3Workspace,D3Options
from .message_plus import conflict_representatives
from .problem import Deadline,DeadlineExceeded,Tree,evaluate
from .solvers import result_dict


@dataclass(frozen=True)
class TerminalOptions:
    cache_entries: int = 20000
    prune: bool = True
    order: str = 'gain'
    small_first: bool = True
    warm_d3: bool = True
    fast_prepare: bool = False
    quotient: bool = False
    private_screen: bool = False
    schedule: str = 'serial'
    similarity_neighbors: int = 0
    cache_bytes: int = 128*1024*1024
    gpu_batch_size: int = 1
    resident_gpu: bool = False
    adaptive_words: bool = True
    lookahead_bound: bool = False
    root_quotient: bool = False
    gpu_native_metadata: bool = False
    gpu_metadata_cache_entries: int = 0
    gpu_sync_tiles: int = 1
    class_capacity_bound: bool = False


PRESETS={'basic':TerminalOptions(cache_entries=0,prune=False,order='natural',small_first=False,warm_d3=False),
         'fast':TerminalOptions()}
PRESETS['prepare']=replace(PRESETS['fast'],fast_prepare=True)
PRESETS['quotient']=replace(PRESETS['prepare'],quotient=True)
PRESETS['screen']=replace(PRESETS['quotient'],private_screen=True)
PRESETS['fifo']=replace(PRESETS['quotient'],order='natural',small_first=False)
PRESETS['small']=replace(PRESETS['quotient'],order='small')
PRESETS['lb']=replace(PRESETS['quotient'],order='lb',schedule='lb')
PRESETS['balanced']=replace(PRESETS['quotient'],schedule='gap')
PRESETS['similar']=replace(PRESETS['quotient'],similarity_neighbors=8)
PRESETS['accelerated']=replace(PRESETS['balanced'],resident_gpu=True,gpu_batch_size=2,
                               lookahead_bound=True,root_quotient=True)


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float
    tree: Tree | None
    exact: bool = False


class TerminalMessages:
    """Terminal factors, cached only when solved exactly; shared uniform costs."""
    def __init__(self,p,backend,threads,options,clock,stats,audit=None,gpu_data_workspace=None,terminal_depth=3):
        if terminal_depth not in (2,3):raise ValueError('Terminal depth must be 2 or 3')
        self.terminal_depth=terminal_depth
        self.p,self.options,self.clock,self.stats,self.audit=p,options,clock,stats,audit
        self.workspace=D3Workspace(p,backend,threads,D3Options(fast_prepare=options.fast_prepare,
            quotient_features=options.quotient,private_screen=options.private_screen,
            gpu_resident=options.resident_gpu,gpu_adaptive_words=options.adaptive_words,
            gpu_native_metadata=options.gpu_native_metadata,
            gpu_metadata_cache_entries=options.gpu_metadata_cache_entries,
            gpu_sync_tiles=options.gpu_sync_tiles))
        if gpu_data_workspace is not None:self.workspace.share_gpu_data_from(gpu_data_workspace)
        self.cache=OrderedDict()
        self.cache_sizes={};self.cache_bytes=0
        self.bound_cache=OrderedDict()
        for key in ('batch_calls','batch_messages','batch_deduplicated','cache_peak_bytes',
                    'lookahead_queries','lookahead_improvements','lookahead_exact','sparse_d3_messages',
                    'lookahead_identity_skips'):
            stats.setdefault(key,0)
        groups=conflict_representatives(p)
        self.conflicts=[(1<<int(row),int(mass)*p._uniform_weight) for row,mass in groups]
        stats['conflict_lower_bound']=sum(value for _,value in self.conflicts)

    def close(self):self.workspace.close()

    def initial(self,rows,allow_cache=True,strengthen=False,terminal_depth=None):
        if len(rows)<self.p.min_leaf:return Interval(math.inf,math.inf,None,True)
        if allow_cache and self.options.cache_entries and rows.mask in self.cache:
            self.cache.move_to_end(rows.mask);self.stats['cache_hits']+=1
            return self.cache[rows.mask]
        counts=None
        if self.options.class_capacity_bound and terminal_depth is not None and len(self.p.labels)>2:
            # Reuse the same counts for STOP and the label-capacity bound.
            counts=[(rows.mask&m).bit_count() for m in self.workspace.class_masks]
            winner=int(np.argmax(counts))
            tree=Tree(label=self.p.labels[winner])
            error=(len(rows)-counts[winner])*self.p._uniform_weight
        else:tree,error=self.workspace.leaf(rows)
        conflict=sum(value for bit,value in self.conflicts if bit&rows.mask)
        lower=min(error,self.p.penalty+conflict)
        if counts is not None:
            # b splits imply at most b+1 predicted labels, with b<=2**D-1.
            # Conflicts and missing labels overlap, hence max rather than sum.
            ranked=sorted(counts,reverse=True);covered=ranked[0];capacity=error
            for b in range(1,min(2**terminal_depth-1,len(ranked)-1)+1):
                covered+=ranked[b]
                missed=(len(rows)-covered)*self.p._uniform_weight
                capacity=min(capacity,b*self.p.penalty+max(conflict,missed))
            previous=lower;lower=max(lower,capacity)
            self.stats['class_capacity_queries']=self.stats.get('class_capacity_queries',0)+1
            self.stats['class_capacity_improvements']=self.stats.get('class_capacity_improvements',0)+int(lower>previous+1e-12)
            self.stats['class_capacity_bound_gain']=self.stats.get('class_capacity_bound_gain',0.)+lower-previous
        lookahead_requested=strengthen and self.options.lookahead_bound and lower<error-1e-12
        # With zero split penalty and no minimum support, conflict masses are
        # additive under every split. One-step lookahead equals the existing LB
        # exactly, so omit this O(F) scan rather than paying for an identical bound.
        identity_bound=self.p.penalty==0 and self.p.min_leaf==0
        if lookahead_requested and identity_bound:self.stats['lookahead_identity_skips']+=1
        if lookahead_requested and not identity_bound:
            # Every non-leaf must pay one split and both child lower bounds.
            # This bound holds for ANY positive remaining depth; a shallow DP
            # optimum itself would be an upper bound and must never be used here.
            self.stats['lookahead_queries']+=1
            previous=lower
            if rows.mask in self.bound_cache:
                lower=max(lower,self.bound_cache[rows.mask]);self.bound_cache.move_to_end(rows.mask)
            else:
                split_lower=math.inf;partitions=set()
                for f in range(self.p.F):
                    self.clock.check()
                    left=self.p.route(rows,f,0)
                    if not left or len(left)==len(rows):continue
                    key=min(left.mask,rows.mask^left.mask)
                    if key in partitions:continue
                    partitions.add(key)
                    right=self.p.route(rows,f,1)
                    children=[self.initial(r,allow_cache=False) for r in (left,right)]
                    split_lower=min(split_lower,self.p.penalty+sum(c.lower for c in children))
                lower=max(lower,min(error,split_lower))
                # Separate bounded namespace: these are depth-independent LBs.
                if self.options.cache_entries and self.options.cache_bytes:
                    self.bound_cache[rows.mask]=lower
                    capacity=min(self.options.cache_entries,(self.options.cache_bytes//4)//max(1,sys.getsizeof(rows.mask)+128))
                    while len(self.bound_cache)>capacity:self.bound_cache.popitem(last=False)
            self.stats['lookahead_improvements']+=int(lower>previous+1e-12)
            self.stats['lookahead_exact']+=int(lower>=error-1e-12)
        if allow_cache and self.options.similarity_neighbors:
            previous=lower
            for mask,record in islice(reversed(self.cache.items()),self.options.similarity_neighbors):
                lower=max(lower,record.lower-(mask&~rows.mask).bit_count()*self.p._uniform_weight)
                self.stats['similarity_comparisons']+=1
            self.stats['similarity_improvements']+=int(lower>previous+1e-12)
            if lower>error+1e-9:raise AssertionError('Similarity lower bound exceeds STOP')
        return Interval(lower,error,tree,lower>=error-1e-12)

    def terminal(self,rows,node,used,kind='terminal'):
        self.clock.check();self.stats['terminal_queries']+=1
        initial=self.initial(rows,strengthen=True,terminal_depth=self.terminal_depth)
        if initial.exact:
            self.stats['terminal_queries_without_oracle']+=1
            if self.audit:self.audit(node,used,rows,initial)
            return initial
        out=self.workspace.solve(rows=rows,node=node,used=used,time_limit=self.clock.remaining(),depth=self.terminal_depth)
        return self._answer(rows,node,used,initial,out,kind)

    def _answer(self,rows,node,used,initial,out,kind):
        prefix='d'+str(self.terminal_depth)
        self.stats[prefix+'_seconds']+=sum(out['timings'].values())
        self.stats[prefix+'_calls']+=1
        self.stats['warm_'+prefix+'_calls']+=int(kind=='warm')
        for key,value in out['timings'].items():
            name=prefix+'_'+key
            self.stats[name]=self.stats.get(name,0.)+value
        self.stats[prefix+'_kernel_calls']+=out['stats']['kernel_calls']
        self.stats['sparse_d3_messages']+=int(out['stats'].get('sparse_words_used',False))
        for key in ('metadata_cache_hits','metadata_sibling_reuse','metadata_native_calls',
                    'gpu_sync_calls','pipeline_prefetch_batches','pipeline_overlap_seconds'):
            self.stats[prefix+'_'+key]=self.stats.get(prefix+'_'+key,0)+out['stats'].get(key,0)
        for key in ('input_features','effective_features','equivalent_features_removed','screened_exact_terminations'):
            name=prefix+'_'+key
            self.stats[name]=self.stats.get(name,0)+out['stats'].get(key,0)
        if out['status']=='OPT':
            self.stats[prefix+'_optimal_calls']+=1
            value=float(out['value']);answer=Interval(value,value,out['tree'],True)
            if value<initial.lower-1e-9:raise AssertionError('Terminal lower bound exceeds its exact cost')
            if self.options.cache_entries:
                # Ancestors are constant on their routed set. Uniform nonnegative
                # costs and shared choices allow contraction of those predicates.
                # D3Workspace removes constant predicates; the cached tree therefore
                # remains legal for any ancestor path inducing this exact RowSet.
                self.cache[rows.mask]=answer;self.cache.move_to_end(rows.mask)
                self.cache_bytes-=self.cache_sizes.get(rows.mask,0)
                # Conservative allowance for a D3 tree, record and dict entry.
                size=sys.getsizeof(rows.mask)+4096
                self.cache_sizes[rows.mask]=size;self.cache_bytes+=size
                byte_limit=self.options.cache_bytes*(3 if self.options.lookahead_bound else 4)//4
                while len(self.cache)>self.options.cache_entries or self.cache_bytes>byte_limit:
                    key,_=self.cache.popitem(last=False)
                    self.cache_bytes-=self.cache_sizes.pop(key);self.stats['cache_evictions']+=1
                self.stats['cache_peak_entries']=max(self.stats['cache_peak_entries'],len(self.cache))
                self.stats['cache_peak_bytes']=max(self.stats['cache_peak_bytes'],self.cache_bytes)
        elif out['status']=='INFEASIBLE':
            answer=Interval(math.inf,math.inf,None,True)
        else:
            upper=min(initial.upper,out['value'])
            answer=Interval(initial.lower,upper,out['tree'] if out['value']<initial.upper else initial.tree,False)
        if self.audit:self.audit(node,used,rows,answer)
        return answer

    def terminal_many(self,requests,kind='terminal'):
        """Deduplicate exact RowSet states before sending bounded GPU batches."""
        if self.workspace.backend!='gpu' or not self.options.resident_gpu or self.options.gpu_batch_size<=1:
            return [self.terminal(rows,node,used,kind) for rows,node,used in requests]
        answers=[None]*len(requests);pending=OrderedDict()
        for i,(rows,node,used) in enumerate(requests):
            self.clock.check();self.stats['terminal_queries']+=1
            initial=self.initial(rows,strengthen=True,terminal_depth=self.terminal_depth)
            if initial.exact:
                answers[i]=initial;self.stats['terminal_queries_without_oracle']+=1
                if self.audit:self.audit(node,used,rows,initial)
            elif rows.mask in pending:
                pending[rows.mask]['indices'].append(i);self.stats['batch_deduplicated']+=1
            else:
                pending[rows.mask]=dict(indices=[i],rows=rows,node=node,used=used,initial=initial)
        jobs=list(pending.values())
        for start in range(0,len(jobs),self.options.gpu_batch_size):
            batch=jobs[start:start+self.options.gpu_batch_size]
            self.stats['batch_calls']+=1;self.stats['batch_messages']+=len(batch)
            outputs=self.workspace.solve_many([dict(rows=r['rows'],node=r['node'],used=r['used']) for r in batch],self.clock.remaining(),depth=self.terminal_depth)
            for r,out in zip(batch,outputs):
                answer=self._answer(r['rows'],r['node'],r['used'],r['initial'],out,kind)
                for i in r['indices']:
                    rows,node,used=requests[i]
                    if i!=r['indices'][0] and answer.tree is not None:
                        value=self.workspace._validate(answer.tree,rows,node,used)
                        if abs(value-answer.upper)>1e-9:raise AssertionError('Deduplicated tree violates its context')
                        if self.audit:self.audit(node,used,rows,answer)
                    answers[i]=answer
        return answers


def solve_terminal_message(p,backend='gpu',time_limit=100,threads=0,options=None,progress=None,audit=None,
                           initial_tree=None,gpu_data_workspace=None):
    """Min-sum on D3 terminal clusters, keeping bounds for all unfinished roots."""
    if p.depth not in (4,5) or p._uniform_weight is None:
        raise ValueError('D3 terminal messages require D4/D5 and uniform weights')
    if not p.early_stop or p.allowed or p.split_costs:
        raise ValueError('D3 terminal messages require STOP and a shared uniform-cost predicate dictionary')
    o=options or (PRESETS['accelerated'] if backend=='gpu' else PRESETS['balanced'])
    if o.cache_entries<0 or o.cache_bytes<0 or o.gpu_batch_size<1 or o.order not in ('natural','gain','small','lb') or o.schedule not in ('serial','lb','gap') or o.similarity_neighbors<0:
        raise ValueError('Invalid terminal-message options')
    if o.similarity_neighbors and p.min_leaf:
        raise ValueError('Similarity transfer requires min_leaf=0 so extending a tree remains feasible')
    clock=Deadline(time_limit)
    name='JT-MP-D3-'+backend.upper()
    stats={'backend':backend,'threads':threads if backend=='cpp' else None,'options':asdict(o),
           'terminal_depth':3,'upper_depth':p.depth-3,'terminal_clusters':2**(p.depth-3),
           'root_tiles_completed':0,'root_features_started':0,'roots_closed_by_bound':0,
           'upper_messages_exact':0,'upper_candidates_evaluated':0,'upper_candidates_pruned':0,
           'second_terminal_calls_avoided':0,'terminal_queries':0,'terminal_queries_without_oracle':0,
           'd3_calls':0,'d3_optimal_calls':0,'d3_kernel_calls':0,'d3_seconds':0.,'warm_d3_calls':0,
           'cache_hits':0,'cache_evictions':0,'cache_peak_entries':0,'setup_seconds':0.,
           'quotient_candidates_removed':0,'constant_candidates_removed':0,
           'similarity_comparisons':0,'similarity_improvements':0,'scheduler_switches':0,
           'initial_incumbent_provided':initial_tree is not None,
           'initial_incumbent_accepted':False,
           'max_columns_applies':False}
    stats['root_equivalent_candidates_removed']=0
    engine=None;roots=[];messages=[];incumbents=[];root_lbs=np.zeros(p.F)
    stop_tree=Tree(label=p.best_label(p.all_rows)) if p.n>=p.min_leaf else None
    stop_value=p.loss(p.all_rows,stop_tree.label) if stop_tree else math.inf
    best_tree,best=stop_tree,stop_value;lb=0.;last_report=0.;active=None
    if stop_tree is not None:
        incumbents.append({'source':'majority_stop','root':None,'seconds':clock.elapsed(),
                           'objective':best,'misclassified':evaluate(p,stop_tree)['misclassified']})

    def accept(tree,value,source,root=None):
        nonlocal best,best_tree
        if tree is not None and value<best-1e-12:
            best,best_tree=value,tree
            metrics=evaluate(p,tree)
            incumbents.append({'source':source,'root':root,'seconds':clock.elapsed(),
                               'objective':value,'misclassified':metrics['misclassified'],
                               'split_nodes':metrics['split_nodes']})
            return True
        return False

    def refresh_side(side):
        if p.depth==5:
            side['lower']=min(side['stop'],min((c['lower'] for c in side['candidates']),default=math.inf))
        side['exact']=side['lower']>=side['upper']-1e-12
        if side['lower']>side['upper']+1e-9:raise AssertionError('Upper message has an invalid interval')

    def refresh_root():
        nonlocal best,best_tree,lb
        if active is not None:
            f,sides=active
            for side in sides:refresh_side(side)
            root_lbs[f]=max(root_lbs[f],p.penalty+sum(s['lower'] for s in sides))
            value=p.penalty+sum(s['upper'] for s in sides)
            if all(s['tree'] is not None for s in sides):
                accept(Tree(feature=f,left=sides[0]['tree'],right=sides[1]['tree']),
                       value,'root_message',f)
        lb=min(stop_value,float(root_lbs.min()))

    def publish(force=False):
        nonlocal last_report
        refresh_root()
        if progress and (force or clock.elapsed()-last_report>=5):
            last_report=clock.elapsed()
            progress(result_dict(name,clock,'RUNNING',best_tree,p,lb,stats=dict(stats),
                                 root_trace=list(roots),message_trace=list(messages),
                                 incumbent_trace=list(incumbents),
                                 message_passing='d3_contracted_junction_tree'))

    def candidate_value(candidate):
        children=candidate['children']
        candidate['lower']=p.penalty+sum(v.lower for v in children)
        candidate['upper']=p.penalty+sum(v.upper for v in children)
        if all(v.tree is not None for v in children):
            return Tree(feature=candidate['feature'],left=children[0].tree,right=children[1].tree)
        return None

    def build_side(f,a):
        # A cached D3 optimum is an upper bound, not a lower bound, for a D4
        # side of a D5 tree. Only terminal D3 states may read exact cache values.
        rows=p.route(p.all_rows,f,a);record=engine.initial(rows,allow_cache=p.depth==4,strengthen=True)
        side={'rows':rows,'lower':record.lower,'upper':record.upper,'tree':record.tree,
              'stop':record.upper,'exact':record.exact,'candidates':[],'initial_exact':record.exact}
        if p.depth==5 and not record.exact:
            partitions=set()
            for g in p.features((a,),(f,)):
                children_rows=[p.route(rows,g,b) for b in (0,1)]
                if o.quotient:
                    if not children_rows[0] or not children_rows[1]:
                        stats['constant_candidates_removed']+=1;continue
                    key=min(r.mask for r in children_rows)
                    if key in partitions:stats['quotient_candidates_removed']+=1;continue
                    partitions.add(key)
                children=[engine.initial(r) for r in children_rows]
                c={'feature':g,'rows':children_rows,'children':children}
                tree=candidate_value(c)
                if tree is not None and c['upper']<side['upper']:
                    side['upper'],side['tree']=c['upper'],tree
                side['candidates'].append(c)
            if o.order=='gain':side['candidates'].sort(key=lambda c:(c['upper'],c['lower'],c['feature']))
            if o.order=='small':side['candidates'].sort(key=lambda c:(min(len(r) for r in c['rows']),c['lower'],c['upper'],c['feature']))
            if o.order=='lb':side['candidates'].sort(key=lambda c:(c['lower'],c['upper'],c['feature']))
            refresh_side(side)
        return side

    def improve(side,value,tree):
        if tree is not None and value<side['upper']:
            side['upper'],side['tree']=value,tree
        refresh_root()

    def solve_side(f,a,sides):
        side,other=sides[a],sides[1-a]
        if side['initial_exact']:return
        if p.depth==4:
            terminal=engine.terminal(side['rows'],(a,),(f,))
            side['lower']=terminal.lower
            improve(side,terminal.upper,terminal.tree)
            if not terminal.exact:raise DeadlineExceeded('D3 terminal message incomplete')
            yield
            return
        for c in side['candidates']:
            clock.check();refresh_root()
            cutoff=min(side['upper'],best-p.penalty-other['lower']) if o.prune else math.inf
            if o.prune and c['lower']>=cutoff:
                stats['upper_candidates_pruned']+=1;continue
            if backend=='gpu' and o.resident_gpu and o.gpu_batch_size>1:
                missing=[b for b in (0,1) if not c['children'][b].exact]
                if len(missing)>1:
                    batch=engine.terminal_many([(c['rows'][b],(a,b),(f,c['feature'])) for b in missing])
                    for b,record in zip(missing,batch):c['children'][b]=record
                    tree=candidate_value(c);improve(side,c['upper'],tree)
                    if not all(v.exact for v in batch):raise DeadlineExceeded('Batched D3 message incomplete')
                    publish();yield
            if o.schedule=='gap':order=sorted((0,1),key=lambda b:-(c['children'][b].upper-c['children'][b].lower))
            elif o.schedule=='lb':order=sorted((0,1),key=lambda b:(c['children'][b].lower,len(c['rows'][b])))
            else:order=sorted((0,1),key=lambda b:len(c['rows'][b])) if o.small_first else (0,1)
            for position,b in enumerate(order):
                if not c['children'][b].exact:
                    c['children'][b]=engine.terminal(c['rows'][b],(a,b),(f,c['feature']))
                    tree=candidate_value(c);improve(side,c['upper'],tree)
                    if not c['children'][b].exact:raise DeadlineExceeded('D3 terminal message incomplete')
                    publish();yield
                cutoff=min(side['upper'],best-p.penalty-other['lower']) if o.prune else math.inf
                if o.prune and position==0 and c['lower']>=cutoff:
                    stats['second_terminal_calls_avoided']+=int(not c['children'][order[1]].exact)
                    break
            stats['upper_candidates_evaluated']+=1
            tree=candidate_value(c);improve(side,c['upper'],tree);publish()
            yield
        refresh_root()

    try:
        clock.check()
        if p.n<p.min_leaf:
            return result_dict(name,clock,'INFEASIBLE',lb=math.inf,stats=stats)
        tick=time.perf_counter()
        engine=TerminalMessages(p,backend,threads,o,clock,stats,audit,gpu_data_workspace)
        if initial_tree is not None:
            initial_value=evaluate(p,initial_tree)['objective']
            if accept(initial_tree,initial_value,'d3_grow'):
                stats['initial_incumbent_accepted']=True
            stats['initial_incumbent_value']=initial_value
        infos=[];root_partitions={}
        for f in range(p.F):
            partition=min(p.route(p.all_rows,f,a).mask for a in (0,1))
            if o.root_quotient and partition in root_partitions:
                # Complement-equivalent roots have identical optimal values;
                # retain one representative in the global min-sum domain.
                root_lbs[f]=math.inf;stats['root_equivalent_candidates_removed']+=1
                continue
            root_partitions[partition]=f
            sides=[engine.initial(p.route(p.all_rows,f,a),allow_cache=p.depth==4) for a in (0,1)]
            root_lbs[f]=p.penalty+sum(v.lower for v in sides)
            infos.append((p.penalty+sum(v.upper for v in sides),f))
        if o.order=='gain':infos.sort()
        if o.order=='small':infos.sort(key=lambda item:(min(len(p.route(p.all_rows,item[1],a)) for a in (0,1)),root_lbs[item[1]],item[0],item[1]))
        if o.order=='lb':infos.sort(key=lambda item:(root_lbs[item[1]],item[0],item[1]))
        stats['root_order']=[f for _,f in infos]
        greedy=greedy_feasible(p,clock)
        if greedy is not None:
            value=engine.workspace._validate(greedy,p.all_rows,(),())
            accept(greedy,value,'greedy')
        stats['setup_seconds']=time.perf_counter()-tick
        publish(True)
        for _,f in infos:
            clock.check();stats['root_features_started']+=1
            if o.prune and root_lbs[f]>=best:
                stats['roots_closed_by_bound']+=1
                roots.append({'root':f,'state':'root_bound','lower':float(root_lbs[f]),
                              'objective':None,'seconds':clock.elapsed(),'best_UB':best})
                stats['root_tiles_completed']+=1;continue
            sides=[build_side(f,a) for a in (0,1)];active=f,sides
            refresh_root()
            if p.depth==4 and backend=='gpu' and o.resident_gpu and o.gpu_batch_size>1 and (not o.prune or root_lbs[f]<best):
                missing=[a for a in (0,1) if not sides[a]['exact']]
                records=engine.terminal_many([(sides[a]['rows'],(a,),(f,)) for a in missing])
                for a,record in zip(missing,records):
                    sides[a]['lower']=record.lower;sides[a]['initial_exact']=record.exact
                    improve(sides[a],record.upper,record.tree)
                if not all(r.exact for r in records):raise DeadlineExceeded('Batched root message incomplete')
            if p.depth==5 and o.warm_d3:
                missing=[a for a in (0,1) if not sides[a]['exact']]
                warms=engine.terminal_many([(sides[a]['rows'],(a,),(f,)) for a in missing],kind='warm')
                for a,warm in zip(missing,warms):
                    improve(sides[a],warm.upper,warm.tree)
                    if not warm.exact:raise DeadlineExceeded('D3 warm start incomplete')
            order=sorted((0,1),key=lambda a:len(sides[a]['rows'])) if o.small_first else (0,1)
            def record_side(a):
                refresh_root()
                messages.append({'root':f,'side':a,'lower':sides[a]['lower'],'upper':sides[a]['upper'],
                                 'exact':sides[a]['exact'],'seconds':clock.elapsed()})
                stats['upper_messages_exact']+=int(sides[a]['exact'])
            if o.schedule=='serial':
                for a in order:
                    if o.prune and root_lbs[f]>=best:break
                    for _ in solve_side(f,a,sides):pass
                    record_side(a)
            else:
                iterators={a:iter(solve_side(f,a,sides)) for a in (0,1)};previous=None
                while iterators:
                    clock.check();refresh_root()
                    if o.prune and root_lbs[f]>=best:break
                    if o.schedule=='gap':a=min(iterators,key=lambda a:(-(sides[a]['upper']-sides[a]['lower']),len(sides[a]['rows']),a))
                    else:a=min(iterators,key=lambda a:(sides[a]['lower'],len(sides[a]['rows']),a))
                    stats['scheduler_switches']+=int(previous is not None and previous!=a);previous=a
                    try:next(iterators[a])
                    except StopIteration:del iterators[a];record_side(a)
            refresh_root()
            exact=all(s['exact'] for s in sides)
            root_upper=p.penalty+sum(s['upper'] for s in sides)
            if not exact and root_lbs[f]<best-1e-9:
                raise AssertionError('Finished root lacks exact or cutoff coverage')
            roots.append({'root':f,'state':'exact' if exact else 'root_bound',
                          'lower':float(root_lbs[f]),'objective':root_upper if exact else None,
                          'seconds':clock.elapsed(),'best_UB':best})
            stats['root_tiles_completed']+=1
            stats['roots_closed_by_bound']+=int(not exact)
            active=None;publish(True)
        refresh_root()
        if lb<best-1e-9:raise AssertionError('Global D3-message lower bound is incomplete')
        return result_dict(name,clock,'OPT',best_tree,p,best,stats=stats,root_trace=roots,message_trace=messages,
                           incumbent_trace=incumbents,
                           message_passing='d3_contracted_junction_tree',optimality_scope='tree_problem')
    except DeadlineExceeded:
        refresh_root()
        return result_dict(name,clock,'TIME',best_tree,p,lb,stats=stats,root_trace=roots,message_trace=messages,
                           incumbent_trace=incumbents,
                           message_passing='d3_contracted_junction_tree',optimality_scope='valid_incumbent_and_message_LBs')
    finally:
        if engine:engine.close()
