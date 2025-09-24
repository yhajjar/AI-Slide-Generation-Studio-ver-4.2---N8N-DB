import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import type { CourseData, GeneratedSlide, GeneralCourseSlide, MicrolearningSlide, KbStatus, AgenticMode, SlideGenState } from './types';
import { CourseType, StructureMethod } from './types';
import { MIN_SLIDES_GENERAL, MIN_SLIDES_MICRO, ALL_GENERAL_CONTENT_TYPES, ALL_MICROLEARNING_CONTENT_TYPES } from './constants';
import { addPage, updatePage } from './glmApi';
import { upsertDocument, retrieveGroundTruth, SlideRequestInfo } from './n8nApi';
import JSZip from 'jszip';

import StepIndicator from './components/StepIndicator';
import Step1_CourseType from './screens/Step1_CourseType';
import Step2_StructureDefinition from './screens/Step2_StructureDefinition';
import Step3_ContentConfiguration from './screens/Step3_ContentConfiguration';
import Step4_PreviewAndGenerate from './screens/Step4_PreviewAndGenerate';
import Step5_Slides from './screens/Step5_Slides';
import { SparklesIcon } from './components/icons/SparklesIcon';
import IconButton from './components/IconButton';
import { CodeBracketIcon } from './components/icons/CodeBracketIcon';
import DebugPanel from './components/DebugPanel';
import Button from './components/Button';
import { ChevronLeftIcon } from './components/icons/ChevronLeftIcon';
import { ChevronRightIcon } from './components/icons/ChevronRightIcon';

const WIZARD_STEPS = [
    "Course Type",
    "Structure",
    "Configuration",
    "Generate",
    "Slides"
];

// --- FOR TESTING ONLY ---
// WARNING: This is a hardcoded API key for temporary testing.
// For production, you MUST use environment variables to keep your key secure.
// const apiKey = process.env.API_KEY!;
const apiKey = '678f5fbf0e0a40e78d6ae129f1ba6f7f.BTae6Sf2vyo0skFq';

const buildStyleGuide = (mode: AgenticMode) => {
  let out = [
    'STYLE GUIDE (applies to all slides)',
    '- Use a clean corporate look with #219ebc as the primary accent.',
    '- Each slide should have a clear heading (e.g., <h2>) and concise body content (<ul> or <p>).',
    '- Output policy: return exactly one final HTML document per slide. No markdown, no code fences, no explanations.',
    ''
  ].join('\n');

  if (mode === 'free') {
    out += [
      'FREE MODE',
      '- You may use reputable external CDNs for CSS/JS/fonts/images.',
      '- Prefer lightweight assets and keep load small.',
      ''
    ].join('\n');
  } else {
    out += [
      'STRICT MODE',
      '- Do NOT use external CSS/JS/fonts/images. Inline CSS only; vanilla JS if needed.',
      '- The document must be fully self-contained and render offline.',
      ''
    ].join('\n');
  }
  return out;
};


const payloadToPrompt = (payload: CourseData, opts: { mode: AgenticMode; groundTruth?: string }): string => {
  const { courseType, courseTopic, slideCount, slides } = payload;
  const { mode, groundTruth } = opts;

  // 1. Global brief
  let out =
    `You are an expert instructional designer. ` +
    `Create a ${slideCount}-slide ${courseType} course titled “${courseTopic}”. ` +
    `Treat the deck as ONE narrative; keep terminology, tone and visuals consistent.\n\n`;

  if (groundTruth) {
      out += `IMPORTANT CONTEXT: You MUST use the following text as the sole source of truth for generating the slides. All facts, figures, and concepts must come from this text. Do not use general knowledge.\n---BEGIN CONTEXT---\n${groundTruth}\n---END CONTEXT---\n\n`;
  } else if (payload.structureMethod === StructureMethod.DOCUMENT && payload.fileName) {
      out +=
      `IMPORTANT CONTEXT: The user uploaded "${payload.fileName}". ` +
      `Use ONLY content derived from this document for facts and figures.\n\n`;
  }

  out += buildStyleGuide(mode) + '\n';

  slides.forEach((slide, i) => {
    const n = i + 1;
    out += `---\nSlide ${n} (${slide.contentType})\n`;

    if (payload.structureMethod === StructureMethod.DOCUMENT) {
      if (slide.autoMode) {
        out += `AI-GENERATE content for this slide grounded in the uploaded document.\n`;
        if (slide.userContent.trim()) out += `Follow these specifics:\n${slide.userContent.trim()}\n`;
      } else {
        out += `Use the following content verbatim:\n${slide.userContent.trim()}\n`;
      }
    } else {
      out += slide.autoMode
        ? `AI-GENERATE content relevant to this slide's topic.\n`
        : `Use the following content verbatim:\n${slide.userContent.trim()}\n`;
    }

    // Interactives: only when requested, no placeholders
    const microSlide = slide as MicrolearningSlide;
    const needsInteractive = courseType === CourseType.MICROLEARNING && Array.isArray(microSlide.interactives) && microSlide.interactives.length > 0;
    
    if (needsInteractive) {
      const list = microSlide.interactives.join(', ');
      out += `Interactive requirement: implement a working ${list} element for this slide (no placeholders).\n`;
    } else {
      out += `No interactive element for this slide.\n`;
    }
  });

  return out;
};

const inlineImagesAsDataURI = async (html: string, onLog: (message: string) => void): Promise<string> => {
    try {
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');
        const images = Array.from(doc.querySelectorAll('img'));
        const imagePromises = images.map(async (img) => {
            const src = img.getAttribute('src');
            if (src && (src.startsWith('http://') || src.startsWith('https://'))) {
                onLog(`[Strict Export] Inlining image: ${src}`);
                try {
                    const response = await fetch(src);
                    if (!response.ok) throw new Error(`Failed to fetch image: ${response.statusText}`);
                    const blob = await response.blob();
                    const reader = new FileReader();
                    const dataUrl = await new Promise<string>((resolve, reject) => {
                        reader.onloadend = () => resolve(reader.result as string);
                        reader.onerror = reject;
                        reader.readAsDataURL(blob);
                    });
                    img.setAttribute('src', dataUrl);
                } catch (e: any) {
                    onLog(`[Strict Export] Warning: Could not inline image ${src}. Error: ${e.message}`);
                }
            }
        });
        await Promise.all(imagePromises);
        return doc.documentElement.outerHTML;
    } catch (e: any) {
        onLog(`[Strict Export] Error during image inlining process: ${e.message}`);
        return html; // Return original HTML on failure
    }
};

const buildSlidesRequestPayload = (slides: (GeneralCourseSlide | MicrolearningSlide)[] | undefined): SlideRequestInfo[] => {
    if (!slides) return [];

    return slides.map(slide => {
        const microSlide = slide as MicrolearningSlide;
        const interactive = (microSlide.interactives && microSlide.interactives.length > 0)
            ? microSlide.interactives[0]
            : 'None';
        
        return {
            id: slide.id,
            contentType: slide.contentType,
            autoMode: slide.autoMode,
            userContent: slide.userContent,
            interactive: interactive,
        };
    });
};


const App: React.FC = () => {
    const [state, setState] = useState<SlideGenState>({
        step: 1,
        courseData: {},
        kbStatus: 'idle',
        kbError: null,
        mode: 'free',
        isLoading: false,
        isRetrievingContent: false,
        isExporting: false,
        generatedSlides: [],
        runId: null,
        glmConversationId: null,
        error: null,
        lastPrompt: '',
        apiLogs: [],
    });
    
    const [showDebug, setShowDebug] = useState(false);
    
    const abortControllerRef = useRef<AbortController | null>(null);
    
    const { step, courseData, kbStatus, kbError, mode, isLoading, isRetrievingContent, isExporting, generatedSlides, runId, glmConversationId, error } = state;

    useEffect(() => {
        if (!state.runId) {
            setState(prev => ({ ...prev, runId: crypto.randomUUID() }));
        }
    }, [state.runId]);


    useEffect(() => {
        return () => {
            if (abortControllerRef.current) {
                abortControllerRef.current.abort();
            }
        };
    }, []);

    const handleUpdateCourseData = useCallback((data: Partial<CourseData>) => {
        setState(prev => ({ ...prev, courseData: { ...prev.courseData, ...data } as Partial<CourseData> }));
    }, []);

    const handleLog = useCallback((message: string | { msg: string; scope?: string; level?: string }) => {
        let logString: string;
        if (typeof message === 'string') {
            logString = message;
        } else {
            const scope = message.scope ? `[${message.scope}]` : '';
            const level = message.level ? `[${message.level.toUpperCase()}]` : '';
            logString = `${scope}${level} ${message.msg}`.trim();
        }
        setState(prev => ({ ...prev, apiLogs: [...prev.apiLogs, `[${new Date().toLocaleTimeString()}] ${logString}`].slice(-500) }));
    }, []);

    const handleFileUpload = useCallback(async (file: File) => {
        if (!runId) {
            handleLog('[Doc Flow] Error: runId is missing.');
            setState(prev => ({
                ...prev,
                kbError: 'A session ID is missing. Please refresh and try again.',
                kbStatus: 'error'
            }));
            return;
        }
        setState(prev => ({...prev, apiLogs: [], kbError: null, kbStatus: 'uploading'}));
        const topic = file.name.split('.').slice(0, -1).join('.').replace(/_/g, ' ');

        handleUpdateCourseData({ fileName: file.name, courseTopic: topic, kbId: undefined });
        handleLog(`[Doc Flow] Starting document processing for: ${file.name} with n8n.`);

        try {
            const { sourceId } = await upsertDocument(
                file,
                { runId, topic },
                handleLog
            );
            
            handleUpdateCourseData({ kbId: sourceId });
            handleLog(`[Doc Flow] Document processed successfully via n8n. Source ID: ${sourceId}`);
            setState(prev => ({...prev, kbStatus: 'ready' }));
        } catch (err: any) {
            handleLog(`[Doc Flow] Error: ${err.message}`);
            setState(prev => ({
                ...prev,
                kbError: err.message || 'An unknown error occurred during document processing.',
                kbStatus: 'error'
            }));
        }
    }, [handleLog, handleUpdateCourseData, runId]);
    
    const handleRetrieveContent = useCallback(async () => {
        if (!runId || !courseData.kbId || !courseData.courseTopic || !courseData.slides || !courseData.courseType) {
            const message = "Missing required data for content retrieval (runId, sourceId, topic, courseType, or slides).";
            setState(prev => ({ ...prev, error: message }));
            handleLog(`[App] Error: ${message}`);
            return;
        }

        setState(prev => ({ ...prev, isRetrievingContent: true, error: null }));
        handleLog('[App] Retrieving auto-generated content from document via n8n...');
        
        const slidesPayload = buildSlidesRequestPayload(courseData.slides as (GeneralCourseSlide | MicrolearningSlide)[]);

        try {
            const items = await retrieveGroundTruth({
                runId,
                sourceId: courseData.kbId,
                topic: courseData.courseTopic,
                courseType: courseData.courseType,
                slides: slidesPayload,
            }, handleLog);
            
            // Map by id to avoid ordering issues
            const contentById = new Map(items.map(it => [it.id, it.SlideContent]));

            setState(prev => {
                const { courseData: currentCourseData } = prev;
                if (!currentCourseData || !currentCourseData.slides) return prev;

                const newSlides = currentCourseData.slides.map((existingSlide, index) => {
                    const slideNumber = index + 1;
                    const retrievedContent = contentById.get(slideNumber);

                    // Only update slides that were in auto-mode and have content from n8n
                    if (retrievedContent && existingSlide.autoMode) {
                        return {
                            ...existingSlide,
                            userContent: retrievedContent,
                        };
                    }
                    return existingSlide;
                });

                return { ...prev, courseData: { ...currentCourseData, slides: newSlides } };
            });

            handleLog('[App] Successfully updated slides with generated content from n8n.');
        } catch (err: any) {
            const message = err.message || 'Failed to retrieve content from n8n.';
            setState(prev => ({ ...prev, error: message }));
            handleLog(`[App] Error retrieving content: ${message}`);
        } finally {
            setState(prev => ({ ...prev, isRetrievingContent: false }));
        }
    }, [courseData, handleLog, runId]);
    
    const handleNext = useCallback(() => {
        if (step === 2 && courseData.structureMethod === StructureMethod.AI) {
            setState(prev => ({ ...prev, step: 4 }));
        } else if (step === 2 && courseData.structureMethod === StructureMethod.DOCUMENT) {
            setState(prev => ({ ...prev, step: 3 }));
        } else {
            setState(prev => ({ ...prev, step: Math.min(prev.step + 1, 5) }));
        }
    }, [step, courseData.structureMethod]);

    const handleBack = useCallback(() => {
        if (step === 4 && courseData.structureMethod === StructureMethod.AI) {
            setState(prev => ({ ...prev, step: 2 }));
        } else if (step === 3 && courseData.structureMethod === StructureMethod.DOCUMENT) {
            setState(prev => ({ ...prev, step: 2 }));
        } else {
            setState(prev => ({ ...prev, step: Math.max(prev.step - 1, 1) }));
        }
    }, [step, courseData.structureMethod]);

    const handleSelectCourseType = (type: CourseType) => {
        if (type === CourseType.GENERAL) {
            handleUpdateCourseData({
                courseType: type,
                courseTopic: '',
                structureMethod: StructureMethod.AI,
                selectedContentTypes: ALL_GENERAL_CONTENT_TYPES,
                slideCount: MIN_SLIDES_GENERAL,
                slides: [],
            });
        } else {
            handleUpdateCourseData({
                courseType: type,
                courseTopic: '',
                structureMethod: StructureMethod.AI,
                slideCount: MIN_SLIDES_MICRO,
                selectedContentTypes: ALL_MICROLEARNING_CONTENT_TYPES,
                slides: [],
            });
        }
        setState(prev => ({ ...prev, kbStatus: 'idle', kbError: null, step: 2 }));
    };
    
    const handleGenerate = useCallback(async () => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
        }
        abortControllerRef.current = new AbortController();
        const controller = abortControllerRef.current;
        
        setState(prev => ({ ...prev, isLoading: true, error: null, generatedSlides: [], glmConversationId: null, apiLogs: [], step: 5 }));

        const generationTimeout = setTimeout(() => {
            handleLog('[App] Generation timed out after 5 minutes.');
            controller.abort();
        }, 300000); // 5 minute timeout

        const onPartial = ({ pos, html, complete }: { pos: number; html: string; complete: boolean }) => {
            setState(s => {
                const slides = [...s.generatedSlides];
                const i = (pos ?? 1) - 1;
                
                while(slides.length <= i) {
                    slides.push({ pageNumber: slides.length + 1, html: '', draft: '', complete: false });
                }

                const prev = slides[i] || { pageNumber: pos, html: '' };
                slides[i] = {
                    ...prev,
                    pageNumber: pos,
                    draft: html,
                    ...(complete ? { html, complete: true } : {}),
                };
                return { ...s, generatedSlides: slides };
            });
        };

        try {
            let groundTruth: string | undefined = undefined;
            if (courseData.structureMethod === StructureMethod.DOCUMENT && courseData.kbId && runId && courseData.courseType) {
                handleLog('[App] Retrieving ground truth from document via n8n...');
                try {
                    const slidesPayload = buildSlidesRequestPayload(courseData.slides as (GeneralCourseSlide | MicrolearningSlide)[]);
                    const items = await retrieveGroundTruth({
                        runId,
                        sourceId: courseData.kbId,
                        topic: courseData.courseTopic!,
                        courseType: courseData.courseType,
                        slides: slidesPayload,
                    }, handleLog);
                    
                    const sortedItems = items.sort((a, b) => a.id - b.id);
                    groundTruth = sortedItems
                        .map(item => `### Slide ${item.id}\n${item.SlideContent}`)
                        .join('\n\n---\n\n');

                } catch (e: any) {
                    clearTimeout(generationTimeout);
                    setState(prev => ({ ...prev, error: `Failed to retrieve context from document via n8n: ${e.message}`, isLoading: false }));
                    return;
                }
            }
            
            const prompt = payloadToPrompt(courseData as CourseData, { mode, groundTruth });
            setState(prev => ({...prev, lastPrompt: prompt}));

            addPage({
                prompt,
                apiKey,
                signal: controller.signal,
                onLog: handleLog,
                onPartial,
                kbId: courseData.kbId,
                onComplete: (convId, slides) => {
                    clearTimeout(generationTimeout);
                    handleLog({
                        level: "info",
                        scope: "render.pipeline",
                        msg: `Received ${slides.length} slides; slide[0] bytes=${slides[0]?.html?.length ?? 0}, doctype=${/^\s*<!doctype/i.test(slides[0]?.html || "")}, hasEscapes=${/\\n|\\"/.test(slides[0]?.html || "")}`
                    });
                    setState(prev => ({...prev, glmConversationId: convId, isLoading: false, generatedSlides: slides }));
                    if (slides.length === 0) {
                        handleLog(`[App] Stream complete, but no slide data was received.`);
                        setState(prev => ({...prev, error: "Generation finished, but the AI did not produce any slides. This might be due to a restrictive prompt or an API issue. Please try modifying your request or starting over."}));
                    } else {
                        handleLog(`[App] Stream complete. ${slides.length} slides were generated successfully via streaming.`);
                    }
                    abortControllerRef.current = null;
                },
                onError: (err) => {
                    clearTimeout(generationTimeout);
                    let errorMessage = err;
                    if (controller.signal.aborted) {
                        errorMessage = "Generation timed out. The server may be busy. Please try again.";
                    }
                    setState(prev => ({...prev, error: errorMessage, isLoading: false }));
                    abortControllerRef.current = null;
                }
            });
        } catch (err: any) {
            clearTimeout(generationTimeout);
            setState(prev => ({...prev, error: err.message, isLoading: false }));
            handleLog(`[App] Error during generation setup: ${err.message}`);
        }
    }, [courseData, handleLog, mode, runId]);
    
    const handleUpdateSlide = useCallback((slideIndex: number, instruction: string) => {
        if (!glmConversationId) {
            setState(prev => ({...prev, error: "Cannot update slide. Missing GLM Conversation ID."}));
            return;
        }

        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
        }
        abortControllerRef.current = new AbortController();
        const controller = abortControllerRef.current;
        setState(prev => ({...prev, isLoading: true, error: null, apiLogs: []}));
        
        const prompt = `Update slide ${slideIndex + 1} with the following revised content/instruction:\n${instruction}`;
        setState(prev => ({...prev, lastPrompt: prompt}));

        const updateTimeout = setTimeout(() => {
            handleLog('[App] Slide update timed out after 2 minutes.');
            controller.abort();
        }, 120000); // 2 minute timeout

        const onPartial = ({ pos, html, complete }: { pos: number; html: string; complete: boolean }) => {
            setState(s => {
                const slides = [...s.generatedSlides];
                const i = (pos ?? 1) - 1;
                if (i >= slides.length) return s;
    
                const prev = slides[i];
                slides[i] = {
                    ...prev,
                    draft: html,
                    ...(complete ? { html, complete: true } : {}),
                };
                return { ...s, generatedSlides: slides };
            });
        };

        updatePage({
            prompt,
            apiKey,
            conversationId: glmConversationId,
            signal: controller.signal,
            onLog: handleLog,
            onPartial,
            kbId: courseData.kbId,
            onComplete: (convId, slides) => {
                clearTimeout(updateTimeout);
                
                handleLog({
                    level: "info",
                    scope: "render.pipeline",
                    msg: `Received ${slides.length} updated slides; slide[0] bytes=${slides[0]?.html?.length ?? 0}, doctype=${/^\s*<!doctype/i.test(slides[0]?.html || "")}, hasEscapes=${/\\n|\\"/.test(slides[0]?.html || "")}`
                });

                setState(prev => {
                    const newSlides = [...prev.generatedSlides];
                    let updated = false;
                    slides.forEach(updatedSlide => {
                        const index = newSlides.findIndex(s => s.pageNumber === updatedSlide.pageNumber);
                        if (index !== -1) {
                            newSlides[index] = { ...updatedSlide, draft: updatedSlide.html, complete: true };
                            updated = true;
                        }
                    });
                     return { ...prev, glmConversationId: convId, isLoading: false, generatedSlides: newSlides };
                });

                if (slides.length === 0) {
                    handleLog(`[App] Slide update stream complete, but no slide data received.`);
                    setState(prev => ({...prev, error: "Update finished, but the AI did not produce a new slide. Please try again."}));
                } else {
                    handleLog(`[App] Slide update stream complete. Changes received successfully.`);
                }
                abortControllerRef.current = null;
            },
            onError: (err) => {
                clearTimeout(updateTimeout);
                let errorMessage = err;
                 if (controller.signal.aborted) {
                    errorMessage = "Slide update timed out. Please try again.";
                }
                setState(prev => ({...prev, error: errorMessage, isLoading: false}));
                abortControllerRef.current = null;
            }
        });
    }, [glmConversationId, handleLog, courseData.kbId]);

    const handleCancelGeneration = useCallback(() => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
            handleLog('[App] User cancelled generation.');
            setState(prev => ({ ...prev, isLoading: false }));
            abortControllerRef.current = null;
        }
    }, [handleLog]);
    
    const validateAndCleanStrictHTML = useCallback((html: string): string => {
        try {
            const parser = new DOMParser();
            const doc = parser.parseFromString(html, 'text/html');
            let cleaned = false;
        
            doc.querySelectorAll('script[src]').forEach(script => {
                handleLog(`[Strict Mode] Removing external script: ${script.getAttribute('src')}`);
                script.remove();
                cleaned = true;
            });

            doc.querySelectorAll('link[rel="stylesheet"]').forEach(link => {
                handleLog(`[Strict Mode] Removing external stylesheet: ${link.getAttribute('href')}`);
                link.remove();
                cleaned = true;
            });
        
            doc.querySelectorAll('img[src]').forEach(img => {
                const src = img.getAttribute('src') || '';
                if (/^https?:\/\//i.test(src) || /^\/\//i.test(src)) {
                     handleLog(`[Strict Mode] Warning: External image found, may not work offline: ${src}`);
                }
            });
        
            return cleaned ? doc.documentElement.outerHTML : html;
        } catch (e) {
            handleLog(`[Strict Mode] Error cleaning HTML: ${e}`);
            return html;
        }
    }, [handleLog]);

    const handleExport = useCallback(async () => {
        if (generatedSlides.length === 0) {
            setState(prev => ({...prev, error: "No slides to export."}));
            return;
        }
        setState(prev => ({...prev, isExporting: true, error: null}));
        
        try {
            const zip = new JSZip();
            let slidesToExport = generatedSlides;

            handleLog(`[App] Starting client-side export in ${mode.toUpperCase()} mode...`);

            if (mode === 'strict') {
                handleLog('[Strict Mode] Preparing slides for offline export...');
                const processedSlides = await Promise.all(
                    generatedSlides.map(async (slide) => {
                        let processedHtml = validateAndCleanStrictHTML(slide.html);
                        processedHtml = await inlineImagesAsDataURI(processedHtml, handleLog);
                        return { ...slide, html: processedHtml };
                    })
                );
                slidesToExport = processedSlides;
            }
            
            slidesToExport.forEach((slide) => {
                const fullHtml = slide.html.includes('<!DOCTYPE html>') 
                    ? slide.html 
                    : `<!DOCTYPE html>
                       <html lang="en">
                         <head>
                           <meta charset="UTF-8">
                           <meta name="viewport" content="width=device-width, initial-scale=1.0">
                           <title>Slide ${slide.pageNumber}</title>
                           ${mode === 'free' ? '<script src="https://cdn.tailwindcss.com?plugins=typography"></script>' : ''}
                         </head>
                         <body>${slide.html}</body>
                       </html>`;
                zip.file(`slide_${slide.pageNumber}.html`, fullHtml);
            });
            
            if (mode === 'strict') {
                const manifest = `<?xml version="1.0" encoding="UTF-8"?>
<manifest identifier="com.ankabutlabs.aigen.course" version="1.2"
          xmlns="http://www.imsglobal.org/xsd/imsccv1p2_imscp_v1p1"
          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
          xsi:schemaLocation="http://www.imsglobal.org/xsd/imsccv1p2_imscp_v1p1 http://www.imsglobal.org/xsd/imscp_v1p1.xsd">
  <organizations default="org">
    <organization identifier="org">
      <title>${courseData.courseTopic || 'Generated Course'}</title>
      <item identifier="item1" identifierref="res1">
        <title>Course Content</title>
      </item>
    </organization>
  </organizations>
  <resources>
    <resource identifier="res1" type="webcontent" href="slide_1.html">
      ${slidesToExport.map(slide => `<file href="slide_${slide.pageNumber}.html"/>`).join('\n      ')}
    </resource>
  </resources>
</manifest>`;
                zip.file('imsmanifest.xml', manifest);
                handleLog('[App] Added imsmanifest.xml for SCORM compatibility.');
            }

            const content = await zip.generateAsync({ type: "blob" });
            
            const link = document.createElement("a");
            link.href = URL.createObjectURL(content);
            const safeTopic = courseData.courseTopic?.replace(/[\W_]+/g,"_") || 'slides';
            link.download = `${safeTopic}.zip`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(link.href);

            handleLog('[App] Export successful.');
        } catch (err: any) {
            const message = err.message || "Failed to create ZIP file.";
            setState(prev => ({...prev, error: message}));
            handleLog(`[App] Export error: ${message}`);
        } finally {
            setState(prev => ({...prev, isExporting: false}));
        }
    }, [generatedSlides, courseData.courseTopic, handleLog, mode, validateAndCleanStrictHTML]);

    const handleStartOver = useCallback(() => {
        const nextStep = courseData.structureMethod === StructureMethod.AI ? 2 : 3;
        setState(prev => ({ 
            ...prev,
            generatedSlides: [],
            glmConversationId: null,
            error: null,
            step: nextStep,
            runId: crypto.randomUUID(), // Regenerate for new run
        }));
    }, [courseData.structureMethod]);
    
    const handleSetCourseData = useCallback((data: Partial<CourseData> | ((prevState: Partial<CourseData>) => Partial<CourseData>)) => {
        setState(prev => {
            const newCourseData = typeof data === 'function' ? data(prev.courseData) : data;
            // FIX: Cast the merged courseData to Partial<CourseData> to resolve a TypeScript error related to discriminated unions.
            // The spread operator was creating a type that TypeScript couldn't correctly infer as a valid member of the CourseData union.
            return { ...prev, courseData: { ...prev.courseData, ...newCourseData } as Partial<CourseData> };
        });
    }, []);

    const isNextDisabled = useMemo(() => {
        if (step === 2) {
            if (!courseData.courseTopic?.trim()) return true;
            if (courseData.structureMethod === StructureMethod.DOCUMENT && kbStatus !== 'ready') return true;
        }
        if (step === 3 && error) {
            return true;
        }
        return false;
    }, [step, courseData, kbStatus, error]);

    const renderStep = () => {
        switch (step) {
            case 1:
                return <Step1_CourseType onSelect={handleSelectCourseType} />;
            case 2:
                return <Step2_StructureDefinition 
                            courseData={courseData as CourseData} 
                            updateCourseData={handleUpdateCourseData}
                            onFileUpload={handleFileUpload}
                            kbStatus={kbStatus}
                            kbError={kbError}
                        />;
            case 3:
                return <Step3_ContentConfiguration 
                            courseData={courseData as CourseData} 
                            setCourseData={handleSetCourseData}
                            onRetrieveContent={handleRetrieveContent}
                            isRetrievingContent={isRetrievingContent}
                        />;
            case 4:
                return <Step4_PreviewAndGenerate 
                            courseData={courseData as CourseData} 
                            onGenerate={handleGenerate} 
                            mode={mode} 
                            setMode={(m) => setState(prev => ({...prev, mode: m}))} 
                        />;
            case 5:
                return <Step5_Slides
                            slides={generatedSlides}
                            isLoading={isLoading}
                            error={error}
                            onUpdateSlide={handleUpdateSlide}
                            onStartOver={handleStartOver}
                            conversationId={glmConversationId}
                            onExport={handleExport}
                            isExporting={isExporting}
                            onCancelGeneration={handleCancelGeneration}
                            mode={mode}
                            onLog={handleLog}
                        />;
            default:
                return <div>Unknown Step</div>;
        }
    };
    
    return (
        <div className="min-h-screen bg-gray-100 text-gray-800 flex flex-col items-center p-4 sm:p-8">
            <div className="w-full max-w-5xl">
                <header className="text-center mb-8">
                    <div className="flex justify-center items-center gap-x-4 gap-y-2 flex-wrap">
                        <h1 className="text-4xl sm:text-5xl font-bold text-gray-900 flex items-center gap-3">
                            <SparklesIcon/> AI Slide Generation Studio
                        </h1>
                        <div className="flex items-center gap-3">
                            <div className="h-7 w-px bg-gray-300"></div>
                            <p className="text-base text-gray-600">Powered by Ankabut Labs</p>
                        </div>
                    </div>
                    <p className="text-lg text-[#219ebc] mt-2">Craft professional courses with the power of AI</p>
                </header>
                
                {step > 0 && step <= WIZARD_STEPS.length && (
                    <div className="mb-8 px-4 py-2 bg-white/50 backdrop-blur-sm rounded-full shadow-inner border border-gray-200">
                        <StepIndicator steps={WIZARD_STEPS} currentStep={step - 1} />
                    </div>
                )}

                <main className="bg-white rounded-xl shadow-lg p-6 sm:p-10 min-h-[500px]">
                    {renderStep()}
                     {step === 3 && error && <p className="text-red-600 mt-4 text-sm text-center font-semibold">{error}</p>}
                </main>
                
                 {step > 1 && step < 5 && (
                    <footer className="mt-8 flex justify-between items-center">
                        <Button onClick={handleBack} variant="secondary" className="flex items-center gap-2">
                            <ChevronLeftIcon className="w-5 h-5"/> Back
                        </Button>
                        {step < 4 && (
                            <Button onClick={handleNext} disabled={isNextDisabled} className="flex items-center gap-2">
                                Next <ChevronRightIcon className="w-5 h-5"/>
                            </Button>
                        )}
                    </footer>
                )}
            </div>
            <div className="fixed bottom-4 left-4 z-50">
                <IconButton onClick={() => setShowDebug(s => !s)} className="bg-gray-700 hover:bg-gray-600">
                    <CodeBracketIcon />
                </IconButton>
            </div>
            {showDebug && (
                <DebugPanel 
                    onClose={() => setShowDebug(false)}
                    state={state}
                />
            )}
        </div>
    );
};

export default App;