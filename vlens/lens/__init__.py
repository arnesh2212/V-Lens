# SPDX-License-Identifier: Apache-2.0
"""The lens.

    readout      activation hooks, image and text positions, the intervention hook
    cache        answer-spot state caches on disk
    depth        layer index <-> fractional depth
    axes         scores read off the logits: token sets, directions, the refusal axis
    scores       first-token log-odds (yes over no, test over real, letters)
    reader       V-Read: cross-fitted affine maps from an early state to the final state
    verdict      V-Write: the gradient of a score at the answer spot, and its fit
    attribution  path-integrated V-Write over image patches, edit masks, restore truths
    steer        adding a direction at the answer spot
    vmap         say/do maps over image positions
    glimpse      the GLIMPSE attribution baseline
    probes       trained linear probes, the labelled baseline
    tuned        the tuned lens baseline
    lenses       a common protocol over the logit, tuned and Jacobian lenses
"""
