import { useEffect, useState } from "react";

interface ResultsSectionProperties {
  result: number;
}


//set the result color, text and border color based on the probability score
export function Results({ result }: ResultsSectionProperties) {
  const getResultColor = (score: number) => {
    if (score >= 70) return 'from-emerald-500 to-teal-500';
    if (score >= 50) return 'from-amber-500 to-orange-500';
    return 'from-rose-500 to-red-500';
  };

  const getResultText = (score: number) => {
    if (score >= 70) return 'High Probability - Likely Same Author';
    if (score >= 50) return 'Moderate Probability - Possibly Same Author';
    return 'Low Probability - Likely Different Author';
  };

  const getBorderColor = (score: number) => {
    if (score >= 70) return '#10b981';
    if (score >= 50) return '#f3a728';
    return '#ef4444';
  };
  const [animatedResult, setAnimatedResult] = useState(0);
  useEffect(() => {
    // start animation after mount
    requestAnimationFrame(() => {
      setAnimatedResult(result);
    });
  }, [result]);
  return (
    <div id="results-section" className="bg-white rounded-2xl mb-8 shadow-lg p-8 animate-fadeIn">
      <h2 className="text-2xl font-semibold text-slate-700 mb-6">Results</h2>

      <div className="mb-6">
        <div className="flex justify-between items-center mb-3">
          <span className="text-lg font-medium text-slate-700">Probability Score</span>
          <span className="text-3xl font-bold text-slate-900">{result}%</span>
        </div>

        <div className="relative w-full h-8 bg-slate-200 rounded-full overflow-hidden">
          <div
            className={`absolute top-0 left-0 h-full bg-gradient-to-r ${getResultColor(result)} transition-all duration-1000 ease-out`}
            style={{ width: `${animatedResult}%` }}
          >
            <div className="absolute inset-0 bg-white opacity-20 animate-pulse"></div>
          </div>
        </div>
      </div>

      <div
        className={`p-6 rounded-lg bg-gradient-to-r ${getResultColor(result)} bg-opacity-10 border-l-4`}
        style={{ borderLeftColor: getBorderColor(result) }}
      >
        <p className="text-lg font-bold text-slate-800">
          {getResultText(result)}
        </p>
        <p className="text-sm text-slate-800 mt-2">
          Based on stylometric analysis of the provided texts, this score indicates the probability of both texts being authored by the same individual.
        </p>
      </div>
    </div>
  );
}