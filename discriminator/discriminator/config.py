from types import SimpleNamespace

Pmain = SimpleNamespace(
    length=512*6,
    seed=0,
)

assert Pmain.length > 0

Pmodel = SimpleNamespace(
    name='discriminator-large',
    patch_size=4,
    d_model=512,
    dim_ff=2048,
    num_heads=8,
    num_layers=8,
    dropout=0.1,
)

Pdata = SimpleNamespace(
    split_ratios=[1, 0.5, 0.2, 0.1, 0.05],
    split_sizes=[100_000, 100_000, 500_000, 500_000, 500_000],
    calibrate_size=100_000,
    calibrate_ratio=0.05,
    save_path='/root/discriminator/.cache/dataset',
)

assert len(Pdata.split_ratios) == len(Pdata.split_sizes)

Ptrain = SimpleNamespace(
    epochs={
        1.0: 1,
        0.5: 1,
        0.2: 1,
        0.1: 2,
        0.05: 2,
    },
    base_lr={
        1.0: 0.0003,
        0.5: 0.0003,
        0.2: 0.0003,
        0.1: 0.0001,
        0.05: 0.0001,
    },
    batch_size=256,
    max_grad_norm=1.0,
    eval_steps=200,
    test_set_size=10_000,
)

# build a general config dict
config          = vars(Pmain)
config['model'] = vars(Pmodel)
config['data']  = vars(Pdata)
config['train'] = vars(Ptrain)
