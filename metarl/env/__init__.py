"""The three-reactor plant with central storage.

    core         Numba kinetics and RK4 integrator
    config       reactors, feed pool, storage tank, specs, task distribution
    layout       observation / action layout and the stage mask
    plant        ``PlantFermentationEnv``: the step, reward and constraints
    faults       the fault registry and ``FaultMixin``
    instruments  the instrument model (``InstrumentMixin``)
    ledger       lots, duals and terminal accounting (``LedgerMixin``)
"""
