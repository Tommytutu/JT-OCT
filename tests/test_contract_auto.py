from jt_oct.contract_cg import ContractOptions,automatic_contract_configuration


def test_small_contract_problem_falls_back_to_cpu():
    selected=automatic_contract_configuration(1372,36,2,5,ContractOptions(),True)
    assert selected['backend']=='cpp'
    assert selected['options'].oracle_batch==16
    assert selected['options'].rmp_every_batches==1
    assert not selected['options'].class_bound


def test_only_validated_binary_profile_delays_rmp():
    fico=automatic_contract_configuration(10459,159,2,5,ContractOptions(),True)
    transactions=automatic_contract_configuration(786363,131,2,5,ContractOptions(),True)
    assert fico['backend']=='gpu' and fico['options'].rmp_every_batches==4
    assert transactions['backend']=='gpu' and transactions['options'].rmp_every_batches==1


def test_multiclass_restores_capacity_bound_and_d4_cadence():
    letter=automatic_contract_configuration(20000,99,26,5,ContractOptions(),True)
    avila=automatic_contract_configuration(20867,85,12,4,ContractOptions(),True)
    assert letter['options'].class_bound and letter['options'].rmp_every_batches==1
    assert avila['options'].class_bound and avila['options'].rmp_every_batches==1


def test_missing_gpu_forces_cpu_without_disabling_safe_defaults():
    selected=automatic_contract_configuration(10459,159,2,5,ContractOptions(),False)
    assert selected['backend']=='cpp' and not selected['options'].resident_gpu
    assert selected['options'].oracle_batch==16
    assert selected['options'].rmp_every_batches==4
    assert selected['options'].state_screen and selected['options'].gpu_fused_join
