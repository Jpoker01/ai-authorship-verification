import {  BookCheck } from 'lucide-react';
import { ChevronDown } from 'lucide-react';

export function AppHeader() {
    {/*function to scroll to instructions section*/}
    const scrollToInstructions = () => {
    const instructionsSection = document.getElementById('instructions-section');
    if (instructionsSection) {
      instructionsSection.scrollIntoView({ behavior: 'smooth' });
    }
  };
    return (
    <div className="text-center mb-12">
      <div className="flex items-center justify-center mb-4 flex-0">
        <BookCheck className="w-28 h-28 md:w-14 md:h-14 lg:w-11 lg:h-11 text-blue-900 mr-3 "/>
        <h1 className="text-5xl font-bold text-slate-900">
          Authorship Verification
        </h1>
      </div>
      <p className="text-xl text-slate-600 max-w-2xl mx-auto my-10">
          Diploma thesis project focused on determining the likelihood of two texts being authored by the same individual.
      </p>
        <button
          onClick={scrollToInstructions}
          className="inline-flex items-center gap-1 text-blue-900 font-medium font-sans hover:text-blue-600 transition-colors"
        >
          Learn more
          <ChevronDown className="w-4 h-4" />
        </button>
    </div>
  );
}