import React, { useEffect, useMemo, useRef, useState } from 'react';
import type { SlideGenState } from '../types';
import LogAnalysisPanel from './LogAnalysisPanel';

interface DebugPanelProps {
    onClose: () => void;
    state: SlideGenState;
}

const DebugPanel: React.FC<DebugPanelProps> = ({ onClose, state }) => {
    const [search, setSearch] = useState("");
    const [autoScroll, setAutoScroll] = useState(true);
    const logBoxRef = useRef<HTMLDivElement>(null);
    const [activeTab, setActiveTab] = useState<'logs' | 'analysis'>('logs');

    const { lastPrompt, apiLogs, ...restOfState } = state;
    
    const sseLast: {runId: string, raw: string} | undefined = typeof window !== 'undefined' ? (window as any).__SSE_LAST__ : undefined;

    const filteredLogs = useMemo(
        () => !search ? apiLogs : apiLogs.filter(l => l.toLowerCase().includes(search.toLowerCase())),
        [apiLogs, search]
    );

    useEffect(() => {
        if (activeTab === 'logs' && autoScroll && logBoxRef.current) {
            logBoxRef.current.scrollTop = logBoxRef.current.scrollHeight;
        }
    }, [filteredLogs, autoScroll, activeTab]);

    return (
        <div className="fixed bottom-4 right-4 w-2/5 max-w-3xl h-3/5 max-h-[700px] bg-gray-800 text-white rounded-lg shadow-2xl z-50 flex flex-col font-mono text-xs">
            <div className="flex justify-between items-center p-3 bg-gray-900 rounded-t-lg flex-shrink-0">
                <h3 className="font-bold text-base">Debug Panel</h3>
                <button onClick={onClose} className="text-xl leading-none px-2 rounded-full hover:bg-gray-700">&times;</button>
            </div>
            <div className="flex-shrink-0 border-b border-gray-700 px-3">
                <nav className="flex space-x-4">
                    <button
                        onClick={() => setActiveTab('logs')}
                        className={`py-2 px-1 text-sm font-medium border-b-2 ${activeTab === 'logs' ? 'border-cyan-400 text-white' : 'border-transparent text-gray-400 hover:text-gray-200'}`}
                    >
                        Raw Logs
                    </button>
                    <button
                        onClick={() => setActiveTab('analysis')}
                        className={`py-2 px-1 text-sm font-medium border-b-2 ${activeTab === 'analysis' ? 'border-cyan-400 text-white' : 'border-transparent text-gray-400 hover:text-gray-200'}`}
                    >
                        Analysis
                    </button>
                </nav>
            </div>

            <div className="flex-grow p-3 overflow-y-auto">
                {activeTab === 'logs' ? (
                    <>
                        <div className="mb-4">
                            <h4 className="font-bold text-[#49b5d1] mb-1">State:</h4>
                            <pre className="bg-gray-900 p-2 rounded whitespace-pre-wrap">{JSON.stringify(restOfState, null, 2)}</pre>
                        </div>
                        <div className="mb-4">
                            <h4 className="font-bold text-[#49b5d1] mb-1">Last Prompt:</h4>
                            <pre className="bg-gray-900 p-2 rounded whitespace-pre-wrap">{lastPrompt || 'No prompt sent yet.'}</pre>
                        </div>
                        <div>
                            <h4 className="font-bold text-[#49b5d1] mb-1">API Logs:</h4>
                            <div className="flex gap-2 items-center my-2">
                                <input
                                    value={search}
                                    onChange={e => setSearch(e.target.value)}
                                    placeholder="filter (e.g. [SSE], phase=answer, tool=add_slide)…"
                                    className="flex-1 w-full bg-gray-700 border border-gray-600 rounded-md py-1 px-2 text-xs"
                                />
                                <label className="flex items-center gap-2 text-xs whitespace-nowrap">
                                    <input type="checkbox" checked={autoScroll} onChange={e => setAutoScroll(e.target.checked)} className="bg-gray-700 border-gray-600 rounded text-[#49b5d1] focus:ring-[#49b5d1]" />
                                    Auto-scroll
                                </label>
                            </div>
                            <div ref={logBoxRef} className="bg-gray-900 p-2 rounded h-60 overflow-y-scroll">
                                {filteredLogs.length > 0 ? filteredLogs.map((log, i) => <div key={i} className="whitespace-pre-wrap">{log}</div>) : 'No logs yet.'}
                            </div>
                        </div>

                        {sseLast?.raw && (
                            <div className="mt-4">
                                <h4 className="font-bold text-[#49b5d1] mb-1">SSE Raw (last run)</h4>
                                <small className="opacity-80">
                                    chars: {sseLast.raw.length.toLocaleString()}
                                    {sseLast.runId ? ` • runId=${sseLast.runId}` : ""}
                                </small>

                                <textarea
                                    readOnly
                                    value={sseLast.raw}
                                    className="w-full h-44 bg-gray-900 border border-gray-700 rounded p-2 mt-1 text-xs"
                                />
                            </div>
                        )}
                    </>
                ) : (
                    <LogAnalysisPanel logs={apiLogs} />
                )}
            </div>
        </div>
    );
};

export default DebugPanel;