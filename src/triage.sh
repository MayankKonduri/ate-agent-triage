#!/usr/bin/env bash
# triage.sh — automated excursion gate for an ATE datalog CSV.
#
# Usage:  ./triage.sh data/defect_4d.csv [outdir]
#
# Runs on every lot as it arrives, before any human or agent is
# involved. Establishes WHETHER to investigate; it does not
# investigate. Prints a verdict to stdout and writes a detailed report
# to <outdir>/gate_<dataset>.txt for the engineer (or agent) who picks
# up the alert.
#
# BASELINE and SIGMA are fixed constants. In production these are
# statistical yield limits derived from accumulated per-lot history --
# AEC-Q002 specifies at least six prior lots, with the limit set a
# fixed number of standard deviations below the historical mean and
# reviewed periodically. That history is outside the scope of this
# study, so the values below stand in for it and are held constant
# across all runs.

set -euo pipefail # Exit immediately if a command fails, also undefined variables are errors, and pipelines fail on the first failed command.

CSV="${1:?usage: triage.sh <file.csv> [outdir]}" # $1 is the first argument, which should be a CSV file. If not provided, it will print the usage message and exit.
OUTDIR="${2:-runs}" # $2 is the second argument, which is optional. If not provided, it defaults to "runs".

BASELINE=94.0      # historical mean part-level yield, %
SIGMA=0.5          # historical lot-to-lot standard deviation, %
K=3                # sigma multiplier
LIMIT=$(awk -v b="$BASELINE" -v s="$SIGMA" -v k="$K" 'BEGIN{printf "%.2f", b-k*s}')

NAME=$(basename "$CSV" .csv)
mkdir -p "$OUTDIR"
REPORT="$OUTDIR/triage_${NAME}.txt"

# ---------------------------------------------------------------- yield
OVERALL=$(awk -F, 'NR>1 { p[$3]; if ($13=="F") b[$3] }
    END { printf "%.2f", 100*(length(p)-length(b))/length(p) }' "$CSV")

VERDICT=$(awk -v y="$OVERALL" -v lim="$LIMIT" 'BEGIN {
    print (y < lim) ? "EXCURSION" : "RELEASE" }')

# ---------------------------------------------------------------- report
{
echo "=================================================================="
echo " AUTOMATED EXCURSION GATE -- LOT DISPOSITION REPORT"
echo "=================================================================="
echo " datalog file      : $CSV"
echo " generated         : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo
echo " YIELD LIMIT DERIVATION"
printf "   historical mean   %.2f %%\n" "$BASELINE"
printf "   historical sigma  %.2f %%\n" "$SIGMA"
printf "   limit (mean-%dsig) %.2f %%\n" "$K" "$LIMIT"
echo
echo "------------------------------------------------------------------"
echo " 1. LOT SUMMARY"
echo "------------------------------------------------------------------"
awk -F, 'NR>1 {
    rows++
    p[$3]; if ($13=="F") bad[$3]
    if ($13=="F") failrows++
    lots[$1]; wf[$1 SUBSEP $2]; tests[$8]
}
END {
    printf "   measurements      %d\n", rows
    printf "   failed rows       %d  (%.2f%%)\n", failrows, 100*failrows/rows
    printf "   parts tested      %d\n", length(p)
    printf "   parts failed      %d\n", length(bad)
    printf "   lots              %d\n", length(lots)
    printf "   wafers            %d\n", length(wf)
    printf "   distinct tests    %d\n", length(tests)
}' "$CSV"
awk -v y="$OVERALL" -v b="$BASELINE" -v s="$SIGMA" 'BEGIN {
    printf "   overall yield     %.2f %%\n", y
    printf "   vs baseline       %+.2f pts  (%+.1f sigma)\n", y-b, (y-b)/s }'
echo
echo "------------------------------------------------------------------"
echo " 2. PER-LOT DISPOSITION"
echo "------------------------------------------------------------------"
printf "   %-6s %9s %9s %9s   %s\n" "LOT" "PARTS" "YIELD" "SIGMA" "STATUS"
awk -F, 'NR>1 {
        key = $1 SUBSEP $3
        if (!(key in seen)) { seen[key]; n[$1]++ }
        if ($13 == "F" && !(key in bad)) { bad[key]; f[$1]++ }
    }
    END { for (l in n) printf "%s %d %.2f\n", l, n[l], 100*(n[l]-f[l])/n[l] }' "$CSV" \
| sort | awk -v lim="$LIMIT" -v b="$BASELINE" -v s="$SIGMA" '
    { st = ($3 < lim) ? "HOLD" : "ok"
      if (st == "HOLD") { held++; hl = hl " " $1 }
      printf "   %-6s %9d %8.2f%% %+8.1f   %s\n", $1, $2, $3, ($3-b)/s, st }
    END { printf "\n   lots held: %d%s\n", held+0, (held ? " --" hl : "") }'
echo
echo "------------------------------------------------------------------"
echo " 3. FAILURE PARETO BY TEST"
echo "------------------------------------------------------------------"
printf "   %-16s %8s %8s %9s  %s\n" "TEST" "FAILS" "RUNS" "RATE" "UNITS"
awk -F, 'NR>1 { n[$8]++; u[$8]=$12; if ($13=="F") f[$8]++ }
    END { for (t in n) printf "%s %d %d %.3f %s\n", t, f[t]+0, n[t], 100*(f[t]+0)/n[t], u[t] }' "$CSV" \
| sort -k4 -rn | awk '{ printf "   %-16s %8d %8d %8.2f%%  %s\n", $1,$2,$3,$4,$5 }'
echo
echo "------------------------------------------------------------------"
echo " 4. HARD BIN SUMMARY"
echo "------------------------------------------------------------------"
echo "   bin 1 = pass   bin 2 = parametric fail   bin 3 = functional fail"
awk -F, 'NR>1 { hb[$14]++ } END { for (b in hb) printf "%s %d\n", b, hb[b] }' "$CSV" \
| sort -n | awk '{ printf "   bin %-4s %10d\n", $1, $2 }'
echo
echo "------------------------------------------------------------------"
echo " 5. VERDICT"
echo "------------------------------------------------------------------"
if [ "$VERDICT" = "EXCURSION" ]; then
cat <<EOT
   STATUS   : EXCURSION
   ACTION   : hold lot group, route to product engineering
   BASIS    : overall part-level yield ${OVERALL}% is below the
              statistical yield limit of ${LIMIT}%
   NOTE     : this gate identifies THAT an excursion occurred. It does
              not identify a root cause. Sections 1-4 are provided as
              starting context only; no dimension beyond lot and test
              has been examined.
EOT
else
cat <<EOT
   STATUS   : RELEASE
   ACTION   : no hold, continue to assembly
   BASIS    : overall part-level yield ${OVERALL}% is at or above the
              statistical yield limit of ${LIMIT}%
EOT
fi
echo "=================================================================="
} | tee "$REPORT"

echo
echo "report written to $REPORT"