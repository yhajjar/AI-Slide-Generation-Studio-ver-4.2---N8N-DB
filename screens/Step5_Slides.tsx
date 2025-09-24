import React, { useState } from 'react';
import type { GeneratedSlide, AgenticMode } from '../types';
import Button from '../components/Button';
import Card from '../components/Card';

interface Step5_SlidesProps {
    slides: GeneratedSlide[];
    isLoading: boolean;
    error: string | null;
    onUpdateSlide: (slideIndex: number, instruction: string) => void;
    onStartOver: () => void;
    conversationId: string | null;
    onExport: () => void;
    isExporting: boolean;
    onCancelGeneration: () => void;
    mode: AgenticMode;
    onLog?: (log: any) => void;
}

const SlideViewer: React.FC<{ slide: GeneratedSlide; mode: AgenticMode; onLog?: (log: any) => void; }> = ({ slide, mode, onLog }) => {
    const [activeTab, setActiveTab] = useState<'Preview' | 'HTML'>('Preview');
    const [copied, setCopied] = useState(false);
    
    const handleIframeInspection = (e: React.SyntheticEvent<HTMLIFrameElement>) => {
        if (!onLog) return;
        try {
            const doc = e.currentTarget.contentDocument!;
            if (!doc || !doc.body) {
                onLog({ ts: Date.now(), level: "warn", scope: "viewer.iframe", msg: `inspect failed for slide ${slide.pageNumber}: no document body.` });
                return;
            };

            const title = doc.title || "(no title)";
            const bodyLen = (doc.body?.innerText || "").length;
            const sheets = Array.from(doc.styleSheets || []).length;
            const icons = doc.querySelectorAll(".material-icons").length;
            
            const probe = doc.createElement("span");
            probe.className = "material-icons";
            probe.textContent = "check";
            doc.body.appendChild(probe);
            const fam = doc.defaultView?.getComputedStyle?.(probe)?.fontFamily || "";
            probe.remove();
            const materialOk = /material icons/i.test(fam);

            const card = doc.querySelector(".flashcard");
            let flipOk = false;
            if (card) {
                (card as HTMLElement).click?.();
                flipOk = card.classList.contains("flipped");
                card.classList.remove("flipped");
            }
            
            onLog({
                ts: Date.now(),
                level: "info",
                scope: "viewer.iframe",
                msg: `Loaded slide ${slide.pageNumber}: title="${title}", bodyChars=${bodyLen}, stylesheets=${sheets}, iconsNodes=${icons}, materialIconsReady=${materialOk}, inlineScriptsWorking=${flipOk}`
            });
        } catch (err: any) {
            onLog({ ts: Date.now(), level: "warn", scope: "viewer.iframe", msg: `inspect failed for slide ${slide.pageNumber}: ${String(err)}` });
        }
    };
    
    const handleIframeLoad = (e: React.SyntheticEvent<HTMLIFrameElement>) => {
        // Run original inspection logic first
        handleIframeInspection(e);

        // Add new scaling logic
        const iframe = e.currentTarget;
        try {
            const doc = iframe.contentDocument;
            if (!doc || !doc.body) return;

            // Define the assumed native resolution of the slide content (e.g., 16:9 presentation)
            const NATIVE_WIDTH = 1280;
            const NATIVE_HEIGHT = 720;

            const { clientWidth: iframeWidth, clientHeight: iframeHeight } = iframe;
            
            // Don't scale if the iframe is already large enough
            if (iframeWidth >= NATIVE_WIDTH && iframeHeight >= NATIVE_HEIGHT) {
                return;
            }

            // Calculate scale factor to fit content within the iframe, maintaining aspect ratio
            const scale = Math.min(iframeWidth / NATIVE_WIDTH, iframeHeight / NATIVE_HEIGHT);
            
            // Apply scaling only if it needs to be scaled down
            if (scale < 1) {
                const body = doc.body;
                // Set body size to the native resolution for consistent scaling
                body.style.width = `${NATIVE_WIDTH}px`;
                body.style.height = `${NATIVE_HEIGHT}px`;
                // Apply the scale transform
                body.style.transform = `scale(${scale})`;
                body.style.transformOrigin = '0 0';

                // Hide scrollbars on the root element of the iframe's document
                if (doc.documentElement) {
                    doc.documentElement.style.overflow = 'hidden';
                }
            }
        } catch (err: any) {
            if (onLog) {
                onLog({ ts: Date.now(), level: "warn", scope: "viewer.iframe.scale", msg: `Failed to apply scaling: ${String(err.message)}` });
            }
        }
    };

    const handleCopy = () => {
        navigator.clipboard.writeText(slide.draft || slide.html || "").then(() => {
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        });
    };

    return (
        <div>
            <div className="flex items-center justify-between border-b border-gray-200 mb-2">
                <div className="flex items-center">
                    <button
                        onClick={() => setActiveTab('Preview')}
                        className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                            activeTab === 'Preview'
                                ? 'border-[#219ebc] text-[#219ebc]'
                                : 'border-transparent text-gray-500 hover:text-gray-700'
                        }`}
                    >
                        Preview
                    </button>
                    <button
                        onClick={() => setActiveTab('HTML')}
                        className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                            activeTab === 'HTML'
                                ? 'border-[#219ebc] text-[#219ebc]'
                                : 'border-transparent text-gray-500 hover:text-gray-700'
                        }`}
                    >
                        HTML
                    </button>
                    {!slide.complete && (
                        <span className="ml-4 text-xs text-amber-600 font-semibold animate-pulse">streaming...</span>
                    )}
                </div>
                 {activeTab === 'HTML' && (
                    <Button variant="secondary" onClick={handleCopy} className="text-xs px-2 py-1">
                        {copied ? 'Copied!' : 'Copy'}
                    </Button>
                )}
            </div>
             {activeTab === 'Preview' ? (
                 <div className="rounded border h-[760px] overflow-hidden bg-white">
                    {slide.complete ? (
                        <iframe
                            className="w-full h-full border-none"
                            title={`Preview of Slide ${slide.pageNumber}`}
                            srcDoc={slide.html}
                            sandbox="allow-scripts allow-same-origin"
                            referrerPolicy="no-referrer"
                            onLoad={handleIframeLoad}
                        />
                    ) : (
                        <div className="w-full h-full flex flex-col items-center justify-center text-sm text-gray-600 bg-gray-50">
                            <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-[#219ebc] mb-4"></div>
                            <p className="font-semibold">Waiting for complete HTML stream...</p>
                            <p className="mt-1 text-xs text-gray-500">({(slide.draft?.length ?? 0).toLocaleString()} bytes received)</p>
                        </div>
                    )}
                </div>
             ) : (
                <div className="bg-gray-900 rounded-md">
                    <textarea
                        readOnly
                        value={slide.draft || slide.html || ""}
                        className="w-full h-[500px] text-xs font-mono p-3 bg-gray-900 text-white border border-gray-700 rounded-md resize-none"
                        spellCheck={false}
                    />
                </div>
             )}
        </div>
    );
};


const Step5_Slides: React.FC<Step5_SlidesProps> = ({
    slides,
    isLoading,
    error,
    onUpdateSlide,
    onStartOver,
    conversationId,
    onExport,
    isExporting,
    onCancelGeneration,
    mode,
    onLog,
}) => {
    const [editingSlide, setEditingSlide] = useState<{ index: number; instruction: string } | null>(null);

    if (isLoading && slides.length === 0) {
        return (
            <div className="flex flex-col items-center justify-center text-center h-full">
                <h2 className="text-3xl font-bold text-gray-900 mb-4">Generating Your Slides...</h2>
                <p className="text-lg text-gray-600">The AI is warming up. Your presentation will be ready shortly.</p>
                <div className="mt-8">
                    <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-[#219ebc]"></div>
                </div>
                <Button onClick={onCancelGeneration} variant="secondary" className="mt-8 bg-red-600 hover:bg-red-700 text-white border-transparent">
                    Cancel Generation
                </Button>
            </div>
        );
    }
    
    if (error) {
        return (
             <div className="flex flex-col items-center justify-center text-center h-full">
                <h2 className="text-3xl font-bold text-red-600 mb-4">An Error Occurred</h2>
                <p className="text-lg text-gray-600">{error}</p>
                <Button onClick={onStartOver} className="mt-8">Try Again</Button>
            </div>
        );
    }

    return (
        <div>
            <div className="flex justify-between items-center mb-6">
                <h2 className="text-3xl font-bold text-gray-900">Generated Slides</h2>
                <div className="flex items-center gap-2">
                    {isLoading ? (
                         <Button onClick={onCancelGeneration} variant="secondary" className="bg-red-500 hover:bg-red-600 text-white">
                            Cancel Generation
                        </Button>
                    ) : (
                        <>
                            <Button
                                onClick={onExport}
                                variant="primary"
                                disabled={isExporting || !conversationId || slides.length === 0}
                            >
                                {isExporting ? 'Exporting...' : 'Download Deck'}
                            </Button>
                            <Button onClick={onStartOver} variant="secondary">
                                Start Over
                            </Button>
                        </>
                    )}
                </div>
            </div>
            
            <div className="space-y-6 max-h-[65vh] overflow-y-auto p-2">
                {slides.map((slide, index) => (
                    <Card key={slide.pageNumber}>
                         <SlideViewer slide={slide} mode={mode} onLog={onLog} />
                         <div className="mt-4 pt-4 border-t border-gray-200">
                            {editingSlide?.index === index ? (
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">Revision Instructions:</label>
                                    <textarea
                                        value={editingSlide.instruction}
                                        onChange={(e) => setEditingSlide({ index, instruction: e.target.value })}
                                        className="w-full bg-white border border-gray-300 rounded-md py-2 px-3 text-gray-900 focus:ring-[#219ebc] focus:border-[#219ebc] min-h-[80px]"
                                        placeholder="e.g., Make the title bolder and add a point about performance."
                                    />
                                    <div className="flex gap-2 mt-2">
                                        <Button 
                                            onClick={() => {
                                                onUpdateSlide(index, editingSlide.instruction);
                                                setEditingSlide(null);
                                            }}
                                            disabled={isLoading}
                                        >
                                            {isLoading ? 'Updating...' : 'Update Slide'}
                                        </Button>
                                        <Button onClick={() => setEditingSlide(null)} variant="secondary">
                                            Cancel
                                        </Button>
                                    </div>
                                </div>
                            ) : (
                                <Button onClick={() => setEditingSlide({ index, instruction: '' })} disabled={isLoading} variant="secondary" className="text-sm px-3 py-1">
                                    Edit Slide
                                </Button>
                            )}
                         </div>
                    </Card>
                ))}
            </div>
        </div>
    );
};

export default Step5_Slides;