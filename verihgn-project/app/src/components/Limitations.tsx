import type { Report } from "../data/schema";
import { validationScope } from "../data/derive";
import { fmtSig, isDefined } from "../data/values";

/** The documented limitations of HEADPATH, every number read from the report. One list, shown on the Statistics
    and Methodology pages and in the footer of every page. */
export function LimitationsList({ report, minP, testId = "limitations" }: { report: Report; minP: number | null; testId?: string }) {
  const scope = validationScope(report);
  const t = report.analysis.primary_test;
  const plan = report.plan;
  return (
    <ul className="bullets" data-testid={testId}>
      <li><strong>No test-set results.</strong> Every performance value comes from the diagnostic validation set: {scope.nGraphs} graphs, {scope.nCrossDesign} from the official
        validation design and {scope.nGraphs - scope.nCrossDesign} held-out graphs of training designs.</li>
      <li><strong>Small number of pairs.</strong> With {t.n_valid_pairs} pairs the exact test reaches p ≤ {t.alpha} only if every pair is discordant in the same direction{isDefined(minP) ? ` (smallest attainable p ${fmtSig(minP, 4)})` : ""}. A non-significant result is not evidence that the arms are equal.</li>
      <li><strong>Diagnostic deviation.</strong> Arm U removes the cell-head clamp during training only; this changes the loss surface as well as the gradient path, so the experiment does not isolate the gradient pathway.</li>
      <li><strong>Short diagnostic runs.</strong> {plan.epochs} epochs ({plan.steps_per_run} optimizer steps) per run; not trained to convergence and not comparable with the paper's results. This is not a reproduction of the paper.</li>
      <li><strong>Adapted pipeline.</strong> The authors' preprocessing is unpublished; features, labels and graph inputs were rebuilt from CircuitNet (adaptations and unresolved assumptions are listed on the Methodology page). This is an adapted reproduction, not an exact one.</li>
      <li><strong>Shared parameters.</strong> Grid-head differences between the arms are confounded by the shared trunk.</li>
      <li><strong>Exploratory values</strong> (per-seed differences) are descriptive only; no inference is drawn from them.</li>
      <li><strong>GPU non-determinism.</strong> Repeating a run gives slightly different values; the test–retest control measures this for one seed per arm.</li>
    </ul>
  );
}
