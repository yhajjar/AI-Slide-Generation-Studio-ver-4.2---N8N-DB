import React, { useState } from 'react';
import type { CourseData, GeneralCourseData, MicrolearningCourseData, KbStatus } from '../types';
import { CourseType, StructureMethod } from '../types';
import { MIN_SLIDES_GENERAL, MAX_SLIDES_GENERAL, MIN_SLIDES_MICRO, MAX_SLIDES_MICRO } from '../constants';
import Card from '../components/Card';
import IconButton from '../components/IconButton';
import Button from '../components/Button';
import { PlusIcon } from '../components/icons/PlusIcon';
import { MinusIcon } from '../components/icons/MinusIcon';

interface Step2Props {
  courseData: CourseData;
  updateCourseData: (data: Partial<CourseData>) => void;
  onFileUpload: (file: File) => void;
  kbStatus: KbStatus;
  kbError: string | null;
}

const NumberControl: React.FC<{
    label: string;
    value: number;
    onIncrement: () => void;
    onDecrement: () => void;
    min: number;
    max: number;
}> = ({ label, value, onIncrement, onDecrement, min, max }) => (
    <div className="flex items-center justify-between w-full">
        <span className="font-medium text-gray-700">{label}</span>
        <div className="flex items-center gap-4">
            <IconButton onClick={onDecrement} disabled={value <= min}>
                <MinusIcon />
            </IconButton>
            <span className="text-xl font-bold text-gray-900 w-8 text-center">{value}</span>
            <IconButton onClick={onIncrement} disabled={value >= max}>
                <PlusIcon />
            </IconButton>
        </div>
    </div>
);

const DocumentUploader: React.FC<{
    onFileUpload: (file: File) => void;
    kbStatus: KbStatus;
    kbError: string | null;
    fileName?: string;
}> = ({ onFileUpload, kbStatus, kbError, fileName }) => {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files && event.target.files[0]) {
            setSelectedFile(event.target.files[0]);
        }
    };

    const handleUploadClick = () => {
        if (selectedFile) {
            onFileUpload(selectedFile);
        }
    };

    const isProcessing = kbStatus !== 'idle' && kbStatus !== 'ready' && kbStatus !== 'error';
    
    let statusMessage = '';
    let messageColor = 'text-gray-600';

    switch(kbStatus) {
        case 'registering':
            statusMessage = 'Creating knowledge base...';
            messageColor = 'text-blue-600';
            break;
        case 'uploading':
            statusMessage = 'Uploading document...';
            messageColor = 'text-blue-600';
            break;
        case 'vectorizing':
            statusMessage = 'Analyzing document content...';
            messageColor = 'text-blue-600';
            break;
        case 'polling':
            statusMessage = 'Finalizing document setup... This may take a moment.';
            messageColor = 'text-blue-600';
            break;
        case 'ready':
            if (fileName) {
                statusMessage = `✅ Ready to use: ${fileName}`;
                messageColor = 'text-green-600';
            }
            break;
        case 'error':
            statusMessage = `Error: ${kbError}`;
            messageColor = 'text-red-600';
            break;
    }


    return (
        <Card className="bg-gray-50">
            <h3 className="text-xl font-bold mb-4 text-center">Document-Based Generation</h3>
            <div className="flex flex-col items-center gap-4">
                <input
                    type="file"
                    onChange={handleFileChange}
                    disabled={isProcessing || kbStatus === 'ready'}
                    className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-[#e3f6fa] file:text-[#219ebc] hover:file:bg-[#cbeff7]"
                    accept=".pdf,.doc,.docx,.txt"
                />
                <Button 
                    onClick={handleUploadClick} 
                    disabled={!selectedFile || isProcessing || kbStatus === 'ready'}
                >
                    {isProcessing ? 'Processing...' : 'Upload and Process'}
                </Button>
                {statusMessage && <p className={`text-sm mt-2 font-medium ${messageColor}`}>{statusMessage}</p>}
            </div>
        </Card>
    );
};


const Step2_StructureDefinition: React.FC<Step2Props> = ({ courseData, updateCourseData, onFileUpload, kbStatus, kbError }) => {
    
    const isGeneral = courseData.courseType === CourseType.GENERAL;
    const generalData = courseData as GeneralCourseData;
    const microData = courseData as MicrolearningCourseData;

    const handleStructureMethodChange = (method: StructureMethod) => {
        updateCourseData({ structureMethod: method });
    };

    const handleSlideTotalChange = (newCount: number) => {
        updateCourseData({ slideCount: newCount });
    };

    return (
        <div className="flex flex-col">
            <h2 className="text-3xl font-bold text-gray-900 mb-2 text-center">Define Course Structure</h2>
            <p className="text-lg text-gray-600 mb-8 text-center">Outline your course, let AI do it, or generate from a document.</p>

            <div className="mb-8">
                <label htmlFor="courseTopic" className="block text-sm font-medium text-gray-700 mb-2">
                    What is the main topic of your course?
                </label>
                <input
                    type="text"
                    id="courseTopic"
                    value={courseData.courseTopic}
                    onChange={(e) => updateCourseData({ courseTopic: e.target.value })}
                    placeholder="e.g., Introduction to Quantum Physics"
                    className="w-full bg-white border border-gray-300 rounded-md py-2 px-4 text-gray-900 focus:ring-[#219ebc] focus:border-[#219ebc]"
                    disabled={courseData.structureMethod === StructureMethod.DOCUMENT && kbStatus === 'ready'}
                />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                <div 
                    onClick={() => handleStructureMethodChange(StructureMethod.AI)}
                    className={`p-4 border-2 rounded-lg cursor-pointer transition-colors text-center ${courseData.structureMethod === StructureMethod.AI ? 'border-[#219ebc] bg-[#e3f6fa]' : 'border-gray-300 hover:border-gray-400'}`}
                >
                    <input type="radio" name="structureMethod" value={StructureMethod.AI} checked={courseData.structureMethod === StructureMethod.AI} className="hidden" readOnly/>
                    <h3 className="text-lg font-bold text-gray-900">Let AI Decide</h3>
                    <p className="text-sm text-gray-600 mt-1">AI generates an optimal structure based on your topic.</p>
                </div>
                <div 
                    onClick={() => handleStructureMethodChange(StructureMethod.USER)}
                    className={`p-4 border-2 rounded-lg cursor-pointer transition-colors text-center ${courseData.structureMethod === StructureMethod.USER ? 'border-[#219ebc] bg-[#e3f6fa]' : 'border-gray-300 hover:border-gray-400'}`}
                >
                     <input type="radio" name="structureMethod" value={StructureMethod.USER} checked={courseData.structureMethod === StructureMethod.USER} className="hidden" readOnly/>
                    <h3 className="text-lg font-bold text-gray-900">Define Manually</h3>
                    <p className="text-sm text-gray-600 mt-1">You have full control over the number of slides.</p>
                </div>
                <div 
                    onClick={() => handleStructureMethodChange(StructureMethod.DOCUMENT)}
                    className={`p-4 border-2 rounded-lg cursor-pointer transition-colors text-center ${courseData.structureMethod === StructureMethod.DOCUMENT ? 'border-[#219ebc] bg-[#e3f6fa]' : 'border-gray-300 hover:border-gray-400'}`}
                >
                     <input type="radio" name="structureMethod" value={StructureMethod.DOCUMENT} checked={courseData.structureMethod === StructureMethod.DOCUMENT} className="hidden" readOnly/>
                    <h3 className="text-lg font-bold text-gray-900">From Document</h3>
                    <p className="text-sm text-gray-600 mt-1">Generate slides based on a file you upload.</p>
                </div>
            </div>

            {(courseData.structureMethod === StructureMethod.USER || courseData.structureMethod === StructureMethod.DOCUMENT) && (
                <Card className="bg-gray-50">
                    <h3 className="text-xl font-bold mb-4 text-center">Structure Builder</h3>
                    {isGeneral ? (
                        <NumberControl
                            label="Total Slides"
                            value={generalData.slideCount}
                            onIncrement={() => handleSlideTotalChange(generalData.slideCount + 1)}
                            onDecrement={() => handleSlideTotalChange(generalData.slideCount - 1)}
                            min={MIN_SLIDES_GENERAL}
                            max={MAX_SLIDES_GENERAL}
                        />
                    ) : (
                         <NumberControl
                            label="Total Slides"
                            value={microData.slideCount}
                            onIncrement={() => handleSlideTotalChange(microData.slideCount + 1)}
                            onDecrement={() => handleSlideTotalChange(microData.slideCount - 1)}
                            min={MIN_SLIDES_MICRO}
                            max={MAX_SLIDES_MICRO}
                        />
                    )}
                </Card>
            )}

            {courseData.structureMethod === StructureMethod.DOCUMENT && (
                 <div className="mt-6">
                    <DocumentUploader
                        onFileUpload={onFileUpload}
                        kbStatus={kbStatus}
                        kbError={kbError}
                        fileName={courseData.fileName}
                    />
                 </div>
            )}
        </div>
    );
};

export default Step2_StructureDefinition;