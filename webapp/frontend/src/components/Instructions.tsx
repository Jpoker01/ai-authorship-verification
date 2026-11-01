import {LucideChevronDown, LucideChevronUp } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface Instruction {
  title: string;
  content: string;
}

interface InstructionsProperties {
  instructions: Instruction[];
  expandedInstructions: number[];
  onToggle: (index: number) => void;
}

export function Instructions({
    //receive an object of type InstructionProperties and pull the three fields out of it
  instructions,
  expandedInstructions,
  onToggle
}: InstructionsProperties) {
  return (
    <div id="instructions-section" className="bg-white rounded-2xl shadow-xl p-8 mb-8">
      <h2 className="text-2xl font-semibold text-slate-700 mb-6">Instructions</h2>
        {/*tailwind helper for vertical spacing between elements*/}
        <div className="space-y-3">
        {instructions.map((instruction, index) => (
          <div
            key={index}
            className="border border-slate-300 rounded-lg overflow-hidden transition-all  hover:border-blue-400"
          >
            <button
              onClick={() => onToggle(index)}
              className="w-full px-6 py-4 flex items-center justify-between text-left bg-slate-50 hover:bg-slate-100 transition-colors"
            >
              <span className="font-medium text-slate-800">
                {instruction.title}
              </span>
              {expandedInstructions.includes(index) ? (
                <LucideChevronUp className="w-5 h-5 text-slate-600" />
              ) : (
                <LucideChevronDown className="w-5 h-5 text-slate-600" />
              )}
            </button>
            <AnimatePresence>
              {expandedInstructions.includes(index) && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ type: 'spring', stiffness: 220, damping: 28}}
                  className="overflow-hidden bg-white"
                >
                  <div className="px-6 py-5">
                    <p className="text-slate-800 leading-relaxed whitespace-pre-wrap">
                      {instruction.content}
                    </p>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

          </div>
        ))}
      </div>
    </div>
  );
}