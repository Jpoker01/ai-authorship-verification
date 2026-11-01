import {CircleX} from 'lucide-react';

interface ErrorSectionProperties {
  error: string;
}

export function Error({ error }: ErrorSectionProperties) {
  return (
<div className="rounded-2xl shadow-lg p-4 mb-8 pb-4 animate-fadeIn bg-red-50">
  <div className="flex items-center gap-4 ml-2 mb-3">
    <CircleX className="w-11 h-11 text-red-500 flex-shrink-0 mt-0.5" />
    <div className="flex-1">
      <h3 className="text-lg font-semibold pb-1.5 text-slate-700">Error</h3>
      <p className="text-md text-slate-800">{error}</p>
    </div>
  </div>
</div>
  )
}
