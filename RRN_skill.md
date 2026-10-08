# RRN skill

**Inspiration:** Neural networks were originally developed to imitate neurons in the brain, and the high accuracy of the golden-ratio frequencies in the RRN supports this. We therefore want to draw inspiration from the brain for how neural networks work.

In a coma the brain shows mainly low-frequency activity, while in wakefulness high-frequency activity increases markedly. Does this mean that high-frequency signals are more important for processing information? We set up two experiments to test this hypothesis.

## Experiment 1: Removing nodes directly

| Band carrying the information | Full network | Lowest 4 nodes removed | Highest 4 nodes removed | Middle 3 nodes removed |
|---|---|---|---|---|
| Low frequency | 89.0% | 53.3% (−35.7) | 89.8% (+0.8) | 89.5% (+0.5) |
| High frequency | 94.0% | 93.5% (−0.5) | 50.3% (−43.7) | 93.3% (−0.7) |

## Experiment 2: Adding an uninformative interfering rhythm, then removing nodes

| Information / interference | Full network | Low-frequency nodes removed | High-frequency nodes removed |
|---|---|---|---|
| Information in low band, interference in high band | 67.9% | 49.6% (−18.3) | 69.7% (+1.8) |
| Information in high band, interference in low band | 74.9% | 74.8% (−0.1) | 49.2% (−25.7) |

## Conclusions

1. **The importance of a frequency band depends on where the information is, not on whether the frequency is high or low.** Removing the group of nodes that carries the information drops accuracy to chance level (−35.7 percentage points for the low band, −43.7 for the high band); moving the information to the other band gives a fully mirrored result.
2. **Nodes can hardly compensate for one another.** The remaining 7 nodes, together with the trainable connections, cannot recover the information of the removed band. This shows that in this model each node is a fairly independent "frequency channel".
3. **Removing an irrelevant band neither hurts nor helps.** In Experiment 1 all changes are within 1 percentage point. In Experiment 2, removing the low-frequency interference nodes gives −0.1 and removing the high-frequency interference nodes gives +1.8, but the latter has a standard error of about 0.9, so the evidence is weak.

In practice, these two experiments are still based on the model itself and cannot truly simulate brain states.

The results show that removing nodes unrelated to the information frequency has almost no effect on the outcome. Does that mean the coupling between nodes is actually not important?

## Experiment 3: Setting W to zero before training

## Experiment 4: Training with connections, setting W to zero at test time (with recalibration)

| Condition | Accuracy | Relative to full network |
|---|---|---|
| Full network (W trained) | 65.3% | — |
| W zeroed before training (classifier only) | 65.5% | +0.1 (17 up, 10 down; no difference) |
| W zeroed after training (nothing else changed) | 54.6% | −10.7 (all 30 runs down) |
| W zeroed after training + classifier recalibrated | 66.8% | +1.4 (24 up, 5 down) |

Zeroing W after training lowers accuracy by 10.7 percentage points, but accuracy recovers completely (66.8%) once the classifier is recalibrated. The drop therefore comes from a mismatch between the classifier and the feature values, not from a loss of information.

**Conclusion:** On the spike data, the nodes can complete the task through their own resonance; cross-frequency coupling is not needed.

**Supplement:** Construct a class of data in which the strength of the high-frequency rhythm follows the phase of the low-frequency rhythm (for example, the high-frequency rhythm appears only at the peak of the low-frequency rhythm), and compare networks with and without cross-frequency connections.
