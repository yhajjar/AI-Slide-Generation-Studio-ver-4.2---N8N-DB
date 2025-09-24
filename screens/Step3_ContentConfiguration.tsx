import React, { useEffect, useCallback } from 'react';
import type { CourseData, GeneralCourseData, GeneralCourseSlide, MicrolearningCourseData, MicrolearningSlide } from '../types';
import { CourseType, GeneralContentType, MicrolearningContentType, StructureMethod } from '../types';
import Card from '../components/Card';
import Button from '../components/Button';
import {
    ALL_GENERAL_CONTENT_TYPES,
    ALL_INTERACTIVE_ELEMENTS,
    ALL_MICROLEARNING_CONTENT_TYPES,
    MAX_SLIDES_GENERAL,
    MIN_SLIDES_GENERAL,
    MAX_SLIDES_MICRO,
    MIN_SLIDES_MICRO
} from '../constants';
import IconButton from '../components/IconButton';
import { TrashIcon } from '../components/icons/TrashIcon';
import { DocumentTextIcon } from '../components/icons/DocumentTextIcon';

interface Step3Props {
  courseData: CourseData;
  setCourseData: React.Dispatch<React.SetStateAction<Partial<CourseData>>>;
  onRetrieveContent: () => void;
  isRetrievingContent: boolean;
}

const Step3_ContentConfiguration: React.FC<Step3Props> = ({ courseData, setCourseData, onRetrieveContent, isRetrievingContent }) => {
    const isGeneral = courseData.courseType === CourseType.GENERAL;
    const isDocumentMode = courseData.structureMethod === StructureMethod.DOCUMENT;
    const generalData = courseData as GeneralCourseData;
    const microData = courseData as MicrolearningCourseData;
    
    const handleGeneralSlideUpdate = useCallback(<K extends keyof GeneralCourseSlide>(
        slideId: number, field: K, value: GeneralCourseSlide[K]
    ) => {
        setCourseData(prev => {
            const currentData = prev as GeneralCourseData;
            const newSlides = currentData.slides.map(slide => 
                slide.id === slideId ? { ...slide, [field]: value } : slide
            );
            return { ...currentData, slides: newSlides };
        });
    }, [setCourseData]);
    
    const handleMicroSlideUpdate = useCallback(<K extends keyof MicrolearningSlide>(
        slideId: number, field: K, value: MicrolearningSlide[K]
    ) => {
        setCourseData(prev => {
            const currentData = prev as MicrolearningCourseData;
            const newSlides = currentData.slides.map(slide => 
                slide.id === slideId ? { ...slide, [field]: value } : slide
            );
            return { ...currentData, slides: newSlides };
        });
    }, [setCourseData]);
    
    useEffect(() => {
        setCourseData(prev => {
            if (!prev.courseType || typeof prev.slideCount === 'undefined') {
                return prev;
            }
    
            const currentSlides = prev.slides || [];
            const targetCount = prev.slideCount;
    
            if (currentSlides.length === targetCount) {
                return prev; // Already in sync
            }
    
            // Handle removing slides
            if (targetCount < currentSlides.length) {
                return { ...prev, slides: currentSlides.slice(0, targetCount) };
            }
    
            // Handle adding slides
            const slidesToAdd: any[] = [];
            const maxId = currentSlides.reduce((max, s) => Math.max(s.id, max), 0);
            const numToAdd = targetCount - currentSlides.length;
    
            if (prev.courseType === CourseType.GENERAL) {
                const currentData = prev as Partial<GeneralCourseData>;
                const slides = (currentData.slides as GeneralCourseSlide[]) || [];
                const singleUseTypes = [GeneralContentType.OVERVIEW, GeneralContentType.OBJECTIVES, GeneralContentType.SUMMARY];
                const usedTypes = new Set(slides.map(s => s.contentType));
                const availableSingleUse = singleUseTypes.filter(t => !usedTypes.has(t));
    
                for (let i = 0; i < numToAdd; i++) {
                    const contentType = availableSingleUse.shift() || GeneralContentType.CONTENT;
                    slidesToAdd.push({
                        id: maxId + i + 1,
                        userContent: '',
                        contentType,
                        autoMode: true,
                    });
                }
                return { ...currentData, slides: [...slides, ...slidesToAdd] };
    
            } else { // Microlearning
                const currentData = prev as Partial<MicrolearningCourseData>;
                const slides = (currentData.slides as MicrolearningSlide[]) || [];
                const singleUseTypes = [MicrolearningContentType.OBJECTIVE, MicrolearningContentType.SUMMARY];
                const usedTypes = new Set(slides.map(s => s.contentType));
                const availableSingleUse = singleUseTypes.filter(t => !usedTypes.has(t));
    
                for (let i = 0; i < numToAdd; i++) {
                    const contentType = availableSingleUse.shift() || MicrolearningContentType.CONTENT;
                    slidesToAdd.push({
                        id: maxId + i + 1,
                        autoMode: false,
                        userContent: '',
                        contentType,
                        interactives: [],
                    });
                }
                return { ...currentData, slides: [...slides, ...slidesToAdd] };
            }
        });
    }, [courseData.courseType, courseData.slideCount, setCourseData]);
    
    const handleAddGeneralSlide = () => {
        setCourseData(prev => {
            const currentData = prev as GeneralCourseData;
            if (currentData.slideCount < MAX_SLIDES_GENERAL) {
                return {
                    ...currentData,
                    slideCount: currentData.slideCount + 1,
                };
            }
            return currentData;
        });
    };

    const handleRemoveGeneralSlideById = (slideId: number) => {
        setCourseData(prev => {
            const currentData = prev as GeneralCourseData;
            if (currentData.slideCount > MIN_SLIDES_GENERAL) {
                return {
                    ...currentData,
                    slideCount: currentData.slideCount - 1,
                    slides: currentData.slides.filter(s => s.id !== slideId)
                };
            }
            return currentData;
        });
    };
    
    const handleAddMicroSlide = () => {
        setCourseData(prev => {
            const currentData = prev as MicrolearningCourseData;
            if (currentData.slideCount < MAX_SLIDES_MICRO) {
                 return {
                    ...currentData,
                    slideCount: currentData.slideCount + 1,
                };
            }
            return currentData;
        });
    };

    const handleRemoveMicroSlideById = (slideId: number) => {
        setCourseData(prev => {
            const currentData = prev as MicrolearningCourseData;
            if (currentData.slideCount > MIN_SLIDES_MICRO) {
                return {
                    ...currentData,
                    slideCount: currentData.slideCount - 1,
                    slides: currentData.slides.filter(s => s.id !== slideId)
                };
            }
            return currentData;
        });
    };

    const getAvailableContentTypes = (currentSlide: GeneralCourseSlide): GeneralContentType[] => {
        const allSlides = (courseData as GeneralCourseData).slides;
        const allPossibleTypes = ALL_GENERAL_CONTENT_TYPES;

        if (allSlides.length <= 1) {
            return allPossibleTypes;
        }

        const singleUseTypes = [GeneralContentType.OVERVIEW, GeneralContentType.OBJECTIVES, GeneralContentType.SUMMARY];
        
        const usedSingleUseTypesByOthers = allSlides
            .filter(slide => slide.id !== currentSlide.id && singleUseTypes.includes(slide.contentType))
            .map(slide => slide.contentType);
            
        const availableTypes = allPossibleTypes.filter(type => {
            if (!singleUseTypes.includes(type)) return true;
            return !usedSingleUseTypesByOthers.includes(type);
        });

        if (!availableTypes.includes(currentSlide.contentType)) {
            return [...availableTypes, currentSlide.contentType].sort();
        }
        
        return availableTypes;
    };

    const getAvailableMicroContentTypes = (currentSlide: MicrolearningSlide): MicrolearningContentType[] => {
        const allSlides = (courseData as MicrolearningCourseData).slides;
        const allPossibleTypes = ALL_MICROLEARNING_CONTENT_TYPES;

        if (allSlides.length <= 1) {
            return allPossibleTypes;
        }

        const singleUseTypes = [MicrolearningContentType.OBJECTIVE, MicrolearningContentType.SUMMARY];

        const usedSingleUseTypesByOthers = allSlides
            .filter(slide => slide.id !== currentSlide.id && singleUseTypes.includes(slide.contentType))
            .map(slide => slide.contentType);
            
        const availableTypes = allPossibleTypes.filter(type => {
            if (!singleUseTypes.includes(type)) return true;
            return !usedSingleUseTypesByOthers.includes(type);
        });

        if (!availableTypes.includes(currentSlide.contentType)) {
            return [...availableTypes, currentSlide.contentType].sort();
        }
        
        return availableTypes;
    };

    if (isGeneral && generalData.slides.length !== generalData.slideCount) return <div>Loading configuration...</div>
    if (!isGeneral && microData.slides.length !== microData.slideCount) return <div>Loading configuration...</div>

    return (
        <div className="flex flex-col">
            <h2 className="text-3xl font-bold text-gray-900 mb-2 text-center">Content Configuration</h2>
            <p className="text-lg text-gray-600 mb-6 text-center">
                Flesh out your slides or mark them for AI generation.
            </p>
            {isDocumentMode && (
                 <div className="flex flex-col items-center gap-4 mb-6">
                    <div className="flex items-center justify-center gap-3 bg-blue-50 border-2 border-blue-200 text-blue-800 rounded-lg p-3 w-full">
                        <DocumentTextIcon className="w-6 h-6 text-blue-600" />
                        <p className="font-semibold">Content generation is powered by your uploaded file: <span className="font-bold">{courseData.fileName}</span></p>
                    </div>
                    <Button onClick={onRetrieveContent} disabled={isRetrievingContent}>
                        {isRetrievingContent ? 'Generating Content...' : 'Auto-Generate Content from Document'}
                    </Button>
                </div>
            )}
            <div className="space-y-6 max-h-[50vh] overflow-y-auto p-2">
                {isGeneral ? (
                     <>
                        <div className="flex justify-center items-center gap-4 mb-4">
                            <Button 
                                variant="primary" 
                                onClick={handleAddGeneralSlide} 
                                disabled={generalData.slideCount >= MAX_SLIDES_GENERAL}
                                className="px-3 py-1 text-sm"
                            >
                                Add Slide
                            </Button>
                            <span className="font-bold text-lg text-gray-700">{generalData.slideCount} / {MAX_SLIDES_GENERAL}</span>
                        </div>
                        {generalData.slides.map((slide, slideIndex) => {
                            const availableTypes = getAvailableContentTypes(slide);
                            return (
                                <Card key={slide.id}>
                                    <div className="flex justify-between items-center mb-4">
                                        <div className="flex items-center gap-2">
                                            <h3 className="text-xl font-bold text-[#219ebc]">Slide {slideIndex + 1}</h3>
                                            <IconButton
                                                onClick={() => handleRemoveGeneralSlideById(slide.id)}
                                                disabled={generalData.slideCount <= MIN_SLIDES_GENERAL}
                                                className="p-1 bg-red-600/80 hover:bg-red-600 text-white disabled:bg-red-300 disabled:hover:bg-red-300"
                                                aria-label={`Remove Slide ${slideIndex + 1}`}
                                            >
                                                <TrashIcon className="w-4 h-4" />
                                            </IconButton>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <label htmlFor={`auto-slide-${slide.id}`} className="text-sm font-medium text-gray-600">Auto</label>
                                            <input id={`auto-slide-${slide.id}`} type="checkbox" checked={slide.autoMode} onChange={() => handleGeneralSlideUpdate(slide.id, 'autoMode', !slide.autoMode)} className="w-4 h-4 text-[#219ebc] bg-gray-100 border-gray-300 rounded focus:ring-[#219ebc]"/>
                                        </div>
                                    </div>
                                    <div className="space-y-4">
                                        <div className="flex justify-between items-center">
                                            <label className="text-sm font-semibold text-gray-700">Content Type</label>
                                            <select
                                                value={slide.contentType}
                                                onChange={(e) => handleGeneralSlideUpdate(slide.id, 'contentType', e.target.value as GeneralContentType)}
                                                className="bg-white border-gray-300 rounded-md py-1 px-2 text-sm"
                                            >
                                               {availableTypes.map(type => <option key={type} value={type}>{type}</option>)}
                                            </select>
                                        </div>
                                        {isDocumentMode ? (
                                            <textarea
                                                placeholder={slide.autoMode ? `AI will generate from the document. Add specific instructions here...` : `Enter manual content for ${slide.contentType}... (will override document)`}
                                                className="w-full bg-white border border-gray-300 rounded-md py-2 px-3 text-gray-900 focus:ring-[#219ebc] focus:border-[#219ebc] min-h-[80px]"
                                                value={slide.userContent}
                                                onChange={(e) => handleGeneralSlideUpdate(slide.id, 'userContent', e.target.value)}
                                            />
                                        ) : (
                                            slide.autoMode ? (
                                                <p className="text-gray-500 text-center py-5 text-sm italic">AI will generate content for this slide based on the selected content type.</p>
                                            ) : (
                                                <textarea
                                                    placeholder={`Enter content for ${slide.contentType}...`}
                                                    className="w-full bg-white border border-gray-300 rounded-md py-2 px-3 text-gray-900 focus:ring-[#219ebc] focus:border-[#219ebc] min-h-[80px]"
                                                    value={slide.userContent}
                                                    onChange={(e) => handleGeneralSlideUpdate(slide.id, 'userContent', e.target.value)}
                                                />
                                            )
                                        )}
                                    </div>
                                </Card>
                            );
                        })}
                    </>
                ) : (
                    <>
                        <div className="flex justify-center items-center gap-4 mb-4">
                            <Button 
                                variant="primary" 
                                onClick={handleAddMicroSlide} 
                                disabled={microData.slideCount >= MAX_SLIDES_MICRO}
                                className="px-3 py-1 text-sm"
                            >
                                Add Slide
                            </Button>
                            <span className="font-bold text-lg text-gray-700">{microData.slideCount} / {MAX_SLIDES_MICRO}</span>
                        </div>
                        {microData.slides.map((slide, slideIndex) => {
                            const availableTypes = getAvailableMicroContentTypes(slide);
                            return (
                             <Card key={slide.id}>
                                 <div className="flex justify-between items-center mb-4">
                                    <div className="flex items-center gap-2">
                                        <h3 className="text-xl font-bold text-[#ffb703]">Slide {slideIndex + 1}</h3>
                                        <IconButton
                                            onClick={() => handleRemoveMicroSlideById(slide.id)}
                                            disabled={microData.slideCount <= MIN_SLIDES_MICRO}
                                            className="p-1 bg-red-600/80 hover:bg-red-600 text-white disabled:bg-red-300 disabled:hover:bg-red-300"
                                            aria-label={`Remove Slide ${slideIndex + 1}`}
                                        >
                                            <TrashIcon className="w-4 h-4" />
                                        </IconButton>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <label htmlFor={`auto-slide-${slide.id}`} className="text-sm font-medium text-gray-600">Auto</label>
                                        <input id={`auto-slide-${slide.id}`} type="checkbox" checked={slide.autoMode} onChange={() => handleMicroSlideUpdate(slide.id, 'autoMode', !slide.autoMode)} className="w-4 h-4 text-[#219ebc] bg-gray-100 border-gray-300 rounded focus:ring-[#219ebc]"/>
                                    </div>
                                 </div>
                                <div className="space-y-4">
                                     <div className="flex justify-between items-center">
                                        <label className="text-sm font-semibold text-gray-700">Content Type</label>
                                        <select
                                            value={slide.contentType}
                                            onChange={(e) => handleMicroSlideUpdate(slide.id, 'contentType', e.target.value as MicrolearningContentType)}
                                            className="bg-white border-gray-300 rounded-md py-1 px-2 text-sm"
                                        >
                                            {availableTypes.map(type => <option key={type} value={type}>{type}</option>)}
                                        </select>
                                    </div>
                                    {isDocumentMode ? (
                                        <textarea
                                            placeholder={slide.autoMode ? `AI will generate from the document. Add specific instructions here...` : `Enter manual content for this slide... (will override document)`}
                                            className="w-full bg-white border border-gray-300 rounded-md py-2 px-3 text-gray-900 focus:ring-[#219ebc] focus:border-[#219ebc] min-h-[100px]"
                                            value={slide.userContent}
                                            onChange={(e) => handleMicroSlideUpdate(slide.id, 'userContent', e.target.value)}
                                        />
                                    ) : (
                                        slide.autoMode ? (
                                            <p className="text-gray-500 text-center py-8 text-sm italic">AI will generate this slide's content and choose an appropriate interactive element.</p>
                                        ) : (
                                            <textarea
                                                placeholder={`Enter content for this slide...`}
                                                className="w-full bg-white border border-gray-300 rounded-md py-2 px-3 text-gray-900 focus:ring-[#219ebc] focus:border-[#219ebc] min-h-[100px]"
                                                value={slide.userContent}
                                                onChange={(e) => handleMicroSlideUpdate(slide.id, 'userContent', e.target.value)}
                                            />
                                        )
                                    )}
                                    <div>
                                        <h5 className="text-sm font-semibold text-gray-700 mb-2">Interactive Element (optional)</h5>
                                        <div className="flex flex-wrap gap-2">
                                            <label key="none-interactive" className="flex items-center gap-2 px-3 py-1 bg-gray-200 rounded-full text-xs cursor-pointer">
                                                <input 
                                                    type="radio"
                                                    name={`interactive-element-${slide.id}`}
                                                    checked={slide.interactives.length === 0}
                                                    onChange={() => {
                                                        handleMicroSlideUpdate(slide.id, 'interactives', []);
                                                    }}
                                                    className="w-3 h-3 text-[#ffb703] bg-white border-gray-300 rounded-full focus:ring-[#ffb703]"
                                                />
                                                None
                                            </label>
                                            {ALL_INTERACTIVE_ELEMENTS.map(interactive => (
                                                <label key={interactive} className="flex items-center gap-2 px-3 py-1 bg-gray-200 rounded-full text-xs cursor-pointer">
                                                    <input 
                                                        type="radio"
                                                        name={`interactive-element-${slide.id}`}
                                                        checked={slide.interactives.includes(interactive)}
                                                        onChange={() => {
                                                            handleMicroSlideUpdate(slide.id, 'interactives', [interactive]);
                                                        }}
                                                        className="w-3 h-3 text-[#ffb703] bg-white border-gray-300 rounded-full focus:ring-[#ffb703]"
                                                    />
                                                    {interactive}
                                                </label>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            </Card>
                        )})}
                    </>
                )}
            </div>
        </div>
    );
};

export default Step3_ContentConfiguration;