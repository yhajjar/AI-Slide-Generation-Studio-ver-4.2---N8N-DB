import React from 'react';
import type { CourseData, AgenticMode } from '../types';
import { StructureMethod } from '../types';
import Button from '../components/Button';
import Card from '../components/Card';

interface Step4Props {
  courseData: CourseData;
  onGenerate: () => void;
  isLoading?: boolean;
  error?: string | null;
  mode: AgenticMode;
  setMode: (mode: AgenticMode) => void;
}

const Step4_PreviewAndGenerate: React.FC<Step4Props> = ({ courseData, onGenerate, isLoading, error, mode, setMode }) => {
    const isAiStructure = courseData.structureMethod === StructureMethod.AI;

    return (
        <div className="flex flex-col">
            <h2 className="text-3xl font-bold text-gray-900 mb-2 text-center">Preview & Generate</h2>
            <p className="text-lg text-gray-600 mb-8 text-center">
                 {isAiStructure 
                    ? "Let's bring your course to life!" 
                    : "Review your configuration and generate your course!"}
            </p>

            <div className="flex flex-col md:flex-row gap-8">
                {!isAiStructure && (
                    <div className="md:w-1/2">
                        <h3 className="text-xl font-bold text-[#219ebc] mb-4">Course Configuration</h3>
                        <Card className="max-h-[400px] overflow-auto bg-gray-50">
                            <pre className="text-xs text-gray-600 whitespace-pre-wrap">
                                {JSON.stringify(courseData, null, 2)}
                            </pre>
                        </Card>
                    </div>
                )}
                <div className={`${isAiStructure ? 'w-full' : 'md:w-1/2'} flex flex-col items-center justify-center`}>
                    <h3 className="text-xl font-bold text-gray-900 mb-4">Generation Mode</h3>
                     <div className="flex gap-4 mb-6">
                        <div onClick={() => setMode('free')} className={`p-3 border-2 rounded-lg cursor-pointer transition-colors text-center ${mode === 'free' ? 'border-[#219ebc] bg-[#e3f6fa]' : 'border-gray-300 hover:border-gray-400'}`}>
                            <h4 className="font-semibold">Free Mode</h4>
                            <p className="text-xs text-gray-600">Allows CDN assets (e.g., Tailwind, Google Fonts). Requires internet.</p>
                        </div>
                        <div onClick={() => setMode('strict')} className={`p-3 border-2 rounded-lg cursor-pointer transition-colors text-center ${mode === 'strict' ? 'border-[#219ebc] bg-[#e3f6fa]' : 'border-gray-300 hover:border-gray-400'}`}>
                             <h4 className="font-semibold">Strict Mode</h4>
                            <p className="text-xs text-gray-600">Self-contained HTML. For offline use or SCORM packages.</p>
                        </div>
                     </div>
                     <h3 className="text-xl font-bold text-gray-900 mb-4">Ready to Create?</h3>
                     <p className="text-gray-600 mb-6">
                        Click the button below to start the AI generation process. Your slides will appear on the next screen.
                     </p>
                     <Button onClick={onGenerate} disabled={isLoading}>
                        {isLoading ? 'Processing...' : 'Generate Slides'}
                    </Button>
                    {error && <p className="text-red-600 mt-4 text-sm">{error}</p>}
                </div>
            </div>
        </div>
    );
};

export default Step4_PreviewAndGenerate;