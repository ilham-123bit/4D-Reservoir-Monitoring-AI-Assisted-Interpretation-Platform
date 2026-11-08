"""Capability ledger: architecture availability does not establish trained predictions."""


def model_catalog():
    return [
        {'family':'Siamese 3D CNN','implementation':'ai/siamese.py; ai/torch_models.py',
         'status':'IMPLEMENTED BUT NOT TRAINED','requirements':'Verified checkpoint, independently justified labels, patch adapter, GPU allocation, overlap reconstruction and input-mask policy.',
         'score':'Raw change-head logit; no prediction is produced by this workflow.',
         'resources':'Scheduled GPU patch inference; full-volume adapter is not eligible.',
         'provenance':'No checkpoint configured or loaded. Checkpoints outside reviewed repository are unknown.'},
        {'family':'Bayesian change','implementation':'ai/detectors.py:BayesianChangeDetector',
         'status':'NOT CONFIGURED','requirements':'Justified scalar null/change distributions and prior; independent calibration for posterior use.',
         'score':'Assumption-based Gaussian log likelihood ratio; numerical fixtures only.',
         'resources':'Bounded CPU scalar maps.','provenance':'No defensible change-class distributions established.'},
        {'family':'Mahalanobis','implementation':'ai/detectors.py:MahalanobisDetector',
         'status':'ELIGIBLE FOR PRELIMINARY FIT','requirements':'Finite named features, fit-only scaling and disjoint unlabeled reference blocks.',
         'score':'Squared multivariate distance; empirical reference cutoff, not probability.',
         'resources':'At most 4096 reference rows and 16384 tile cells.','provenance':'Fit provenance and spatial block hashes recorded per run.'},
        {'family':'Autoencoder','implementation':'ai/detectors.py:LinearAutoencoderDetector; ai/torch_models.py',
         'status':'PCA ELIGIBLE; CNN NOT TRAINED','requirements':'PCA: finite scaled features with nonzero variance; CNN: justified normal/reference training population and checkpoint.',
         'score':'PCA feature reconstruction residual. No nonlinear CNN prediction is emitted.',
         'resources':'Bounded CPU PCA; CNN requires scheduled GPU training/inference.',
         'provenance':'PCA fit reference recorded; no CNN checkpoint configured.'},
        {'family':'Unsupervised','implementation':'ai/detectors.py:IsolationForestDetector, GaussianMixtureDetector',
         'status':'ELIGIBLE WITH SCIKIT-LEARN','requirements':'Finite scaled reference features, deterministic seed, threshold subset; mixture convergence checked.',
         'score':'Isolation score / negative log likelihood; uncalibrated novelty, no geological class.',
         'resources':'40 trees by default, maximum 100; two mixture components; one CPU thread.',
         'provenance':'Software versions, input digest, parameters, reference hashes and thresholds recorded.'},
    ]
