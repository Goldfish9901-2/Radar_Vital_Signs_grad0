"""Frozen single-seed FTU TCN protocol and label-independent donor mapping."""
import numpy as np

PROTOCOL = dict(seed=42, max_epochs=30, patience=8, batch_size=64, num_workers=0,
    lr=3e-4, weight_decay=1e-4, dropout=.15, hidden_channels=48, num_blocks=4,
    kernel_size=7, model="tcn", time_channels=1, freq_channels=1,
    checkpoint_selection="minimum validation MAE; never test or tracking metrics",
    train_subjects=[2,3,4,5,7,9], val_subjects=[6,10], test_subjects=[1,8],
    negative_control="swap p01/p08 matching record suffix and window_start; both branches together",
    dynamic_metrics="within-record Pearson r and std(pred)/std(HR); pooled record-centered metrics",
    limitations=["single seed", "two previously explored test subjects", "eight overlapping windows per record",
                 "offset zero assumed, not calibrated synchronization"])


def donor_indices(frame):
    """Bijective cross-subject swap; only identities enter this function."""
    frame = frame.reset_index(drop=True)
    assert set(frame.group_key) == {"FTU/participant/1", "FTU/participant/8"}
    lookup = {(r.sample_tag,int(r.window_start)): i for i,r in enumerate(frame.itertuples())}
    assert len(lookup) == len(frame)
    donors=[]
    for row in frame.itertuples():
        prefix, suffix=row.sample_tag.split('_',1)
        other={'p01':'p08','p08':'p01'}[prefix]
        donors.append(lookup[(other+'_'+suffix,int(row.window_start))])
    donors=np.asarray(donors,dtype=int)
    assert len(np.unique(donors))==len(frame)
    assert (frame.group_key.to_numpy()!=frame.group_key.to_numpy()[donors]).all()
    return donors
