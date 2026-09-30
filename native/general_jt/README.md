# General JT source provenance

These four headers were copied on 2026-09-30 from the user's sibling project
`D:/论文高/General JT/JT-CM/native`. They are vendored so this repository does
not need that project at runtime. The native DLL is built from
`../accuracy_oracle.cpp`; the original CC-F1 coordinator is not used.

- `cm5_common.hpp`: binary row masks, data and concrete tree witnesses.
- `cm_weighted_support.hpp`: common scalar support types.
- `cm_mcc_gpu.hpp`: scalar D3 evaluator, compact transposed data, equivalent
  predicates, symmetric pairs, integer evaluation, device reduction and
  background geometry preparation.
- `cm_shallow_linear.hpp`: exact D2 scalar evaluator (tested as a service;
  the public accelerated D4/D5 CG profile uses D3 tails).

Only `cm_mcc_gpu.hpp` differs from the copied upstream version: it adds optional
suppression of unused conditional-root downloads and the final redundant device
synchronization, plus a cancellation flag for background preparation. Default
flags retain upstream behavior; the accuracy wrapper opts into the former two.
The final blocking device-to-host witness copy still waits for the computation.

Source SHA256 values and the compiled DLL hash are recorded in
`provenance.json`. Local timings after the synchronization change must combine
kernel-launch and join time: GPU execution is now charged largely to the
blocking join. Asynchronous prefill time overlaps these timers and must not
be added to wall time.
