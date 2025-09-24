import React from 'react';
import Card from '../components/Card';
import { CourseType } from '../types';

interface Step1_CourseTypeProps {
  onSelect: (type: CourseType) => void;
}

const Step1_CourseType: React.FC<Step1_CourseTypeProps> = ({ onSelect }) => {
  return (
    <div className="flex flex-col items-center">
      <h2 className="text-3xl font-bold text-gray-900 mb-2">Choose Your Course Type</h2>
      <p className="text-lg text-gray-600 mb-8">Select a foundation for your new course.</p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 w-full max-w-3xl">
        <Card onClick={() => onSelect(CourseType.GENERAL)} className="text-center">
          <h3 className="text-2xl font-bold text-[#219ebc] mb-3">General Course</h3>
          <p className="text-gray-700">A comprehensive, module-based structure for in-depth learning. Ideal for structured, longer-form educational content.</p>
          <ul className="text-left mt-4 text-gray-500 space-y-1 text-sm list-disc list-inside">
              <li>Organized into Modules & Slides</li>
              <li>Covers broader topics</li>
              <li>Focuses on foundational knowledge</li>
          </ul>
        </Card>
        <Card onClick={() => onSelect(CourseType.MICROLEARNING)} className="text-center">
          <h3 className="text-2xl font-bold text-[#ffb703] mb-3">Microlearning Course</h3>
          <p className="text-gray-700">Short, focused, and engaging slide-based content. Perfect for quick lessons and interactive skill-building.</p>
          <ul className="text-left mt-4 text-gray-500 space-y-1 text-sm list-disc list-inside">
              <li>Max 10 slides for brevity</li>
              <li>Includes interactive elements</li>
              <li>Designed for rapid knowledge transfer</li>
          </ul>
        </Card>
      </div>
    </div>
  );
};

export default Step1_CourseType;