import { MIN_CHAR_COUNT, MAX_CHAR_COUNT } from '../services/config';

interface TextVerificationProperties{
  text1: string;
  text2: string;
  onText1Change: (text: string) => void;
  onText2Change: (text: string) => void;
  isAnalyzing: boolean;
  onAnalyze: () => void;
}

export function TextVerification({
  text1,
  text2,
  onText1Change,
  onText2Change,
  isAnalyzing,
  onAnalyze
}: TextVerificationProperties) {
  const text1Length = text1.trim().length;
  const text2Length = text2.trim().length;

  // Check if texts meet the requirements
  const isText1Valid = text1Length >= MIN_CHAR_COUNT && text1Length <= MAX_CHAR_COUNT;
  const isText2Valid = text2Length >= MIN_CHAR_COUNT && text2Length <= MAX_CHAR_COUNT;
  const canAnalyze = isText1Valid && isText2Valid && !isAnalyzing;

  const getCharCountClass = (length: number) => {
    if (length === 0) return 'text-slate-700';
    if (length < MIN_CHAR_COUNT || length > MAX_CHAR_COUNT) return 'text-red-600';
    return 'text-green-600';
  };

  const getCharCountMessage = (length: number) => {
    if (length < MIN_CHAR_COUNT) {
      return `${length.toLocaleString()} / ${MIN_CHAR_COUNT.toLocaleString()} characters`;
    }
    if (length > MAX_CHAR_COUNT) {
      return `${length.toLocaleString()} / ${MAX_CHAR_COUNT.toLocaleString()} characters`;
    }
    return `${length.toLocaleString()} characters ✓`;
  };

  return (
    <div className="bg-white rounded-2xl shadow-lg p-8 mb-10">
      <h2 className="text-2xl font-semibold text-slate-700 mb-6">Text Input</h2>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <div>
          <label className="block text-md font-medium text-slate-700 mb-3">
            Text 1
          </label>
          <textarea
            value={text1}
            onChange={(e) => onText1Change(e.target.value)}
            placeholder="Paste the first text here..."
            className={"w-full h-80 px-4 py-4 border rounded-lg hover:border-blue-400 focus:ring-inset focus:ring-2 focus:ring-blue-400 resize-none transition-shadow"}
          />
          <div className={`mt-2 text-md font-semibold ${getCharCountClass(text1Length)}`}>
            {getCharCountMessage(text1Length)}
          </div>
        </div>

        <div>
          <label className="block text-md font-medium text-slate-700 mb-3">
            Text 2
          </label>
          <textarea
            value={text2}
            onChange={(e) => onText2Change(e.target.value)}
            placeholder="Paste the second text here..."
            className={"w-full h-80 px-4 py-3 border rounded-lg hover:border-blue-400 focus:ring-inset focus:ring-2 focus:ring-blue-400 resize-none transition-shadow"}
          />
          <div className={`mt-2 text-md font-semibold ${getCharCountClass(text2Length)}`}>
            {getCharCountMessage(text2Length)}
          </div>
        </div>
      </div>

      <div className="flex flex-col items-center gap-2">
        <button
          onClick={() => {
            onAnalyze();
          }}
          disabled={!canAnalyze}
          className="px-8 py-4 bg-gradient-to-r from-blue-600 to-cyan-500 text-white font-semibold rounded-lg shadow-xl hover:from-blue-700 hover:to-cyan-600 disabled:from-slate-400 disabled:to-slate-400 disabled:cursor-not-allowed transition-all"
        >
          {isAnalyzing ? (
            <span className="flex items-center">
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Analyzing...
            </span>
          ) : (
            'Verify Authorship'
          )}
        </button>
      </div>
    </div>
  );
}