import React, { useMemo } from 'react';
import { CheckCircleIcon } from './icons/CheckCircleIcon';
import { XCircleIcon } from './icons/XCircleIcon';
import { InformationCircleIcon } from './icons/InformationCircleIcon';
import { ClockIcon } from './icons/ClockIcon';
import { ArrowUpCircleIcon } from './icons/ArrowUpCircleIcon';
import { ArrowDownCircleIcon } from './icons/ArrowDownCircleIcon';

interface LogAnalysisPanelProps {
    logs: string[];
}

interface HttpInfo {
  method: string;
  url: string;
  status?: number;
  statusText?: string;
  duration?: number;
  ray?: string;
  headers?: string;
  body?: string;
  preview?: string;
}

interface AnalysisPhase {
  key: string;
  name: string;
  status: 'pending' | 'success' | 'error';
  events: { type: 'info' | 'error', message: string }[];
  request?: HttpInfo;
  response?: HttpInfo;
  data: Record<string, string | number>;
  isPolling?: boolean;
  pollingAttempts?: HttpInfo[];
}

const parseLogsForAnalysis = (logs: string[]): AnalysisPhase[] => {
    const phases: AnalysisPhase[] = [];
    let currentPhase: AnalysisPhase | null = null;
    let netBuffer: { type: 'req' | 'res' | null, lines: string[] } = { type: null, lines: [] };

    const finalizeNetBuffer = () => {
        if (!currentPhase || !netBuffer.type) return;
        const fullText = netBuffer.lines.join('\n');
        
        if (netBuffer.type === 'req') {
            const match = fullText.match(/▶\s+(\w+)\s+(https\S+)/);
            if (match) {
                currentPhase.request = {
                    method: match[1],
                    url: match[2],
                    body: fullText.includes('Body:') ? fullText.split('Body:')[1].trim() : undefined,
                    headers: fullText.includes('Headers:') ? fullText.split('Headers:')[1].split('Body:')[0].trim() : undefined,
                };
            }
        } else if (netBuffer.type === 'res') {
            const match = fullText.match(/◀\s+(\w+)\s+(https\S+)\s+→\s+(\d+)\s+([^(\n]+)\s+\((\d+)\s+ms\)\s+ray=([\w-]+)/);
            const info: HttpInfo = {
                method: match?.[1] ?? 'N/A',
                url: match?.[2] ?? 'N/A',
                status: match ? parseInt(match[3], 10) : undefined,
                statusText: match?.[4].trim() ?? 'N/A',
                duration: match ? parseInt(match[5], 10) : undefined,
                ray: match?.[6] ?? 'N/A',
                headers: fullText.includes('Response Headers:') ? fullText.split('Response Headers:')[1].split('Response Preview:')[0].trim() : undefined,
                preview: fullText.includes('Response Preview:') ? fullText.split('Response Preview:')[1].trim() : undefined,
            };

            if (currentPhase.isPolling) {
                currentPhase.pollingAttempts = currentPhase.pollingAttempts || [];
                currentPhase.pollingAttempts.push(info);
            } else {
                currentPhase.response = info;
            }

            if (info.status && info.status >= 400) {
                currentPhase.status = 'error';
            }
            
            if (info.preview) {
                 try {
                    const json = JSON.parse(info.preview);
                    if (json?.data?.id) currentPhase.data.kbId = json.data.id;
                    if (json?.data?.successInfos?.[0]?.documentId) currentPhase.data.docId = json.data.successInfos[0].documentId;
                 } catch {}
            }
        }
        netBuffer = { type: null, lines: [] };
    };

    logs.forEach(log => {
        const message = log.substring(log.indexOf('] ', log.indexOf('] ') + 1) + 2);

        if (message.startsWith('[KB Create] Creating')) {
            finalizeNetBuffer();
            currentPhase = { key: `create-kb-${phases.length}`, name: '1. Create Knowledge Base', status: 'pending', events: [], data: {} };
            phases.push(currentPhase);
        } else if (message.startsWith('[KB Upload] Uploading')) {
            finalizeNetBuffer();
            currentPhase = { key: `upload-${phases.length}`, name: '2. Upload Document', status: 'pending', events: [], data: {} };
            phases.push(currentPhase);
        } else if (message.startsWith('[KB Vectorize] Triggering')) {
            finalizeNetBuffer();
            currentPhase = { key: `vectorize-${phases.length}`, name: '3. Trigger Vectorization', status: 'pending', events: [], data: {} };
            phases.push(currentPhase);
        } else if (message.startsWith('[Doc Status] Starting status polling')) {
            finalizeNetBuffer();
            currentPhase = { key: `poll-${phases.length}`, name: '4. Poll Document Status', status: 'pending', events: [], data: {}, isPolling: true };
            phases.push(currentPhase);
        } else if (message.startsWith('[GLM API] Sending request')) {
            finalizeNetBuffer();
            currentPhase = { key: `generate-${phases.length}`, name: 'Generate Slides', status: 'pending', events: [], data: {} };
            phases.push(currentPhase);
        }

        if (message.startsWith('[NET]')) {
            const isReq = message.includes('▶');
            const isRes = message.includes('◀');
            const currentType = isReq ? 'req' : (isRes ? 'res' : null);

            if (netBuffer.type && netBuffer.type !== currentType) {
                finalizeNetBuffer();
            }

            netBuffer.type = currentType;
            if (netBuffer.type) {
                netBuffer.lines.push(message.substring(message.indexOf(']') + 2));
            }

        } else if (currentPhase) {
            const isError = /error|fail/i.test(message);
            currentPhase.events.push({ type: isError ? 'error' : 'info', message });
            if (isError) currentPhase.status = 'error';
            
            if (message.includes('Successfully created KB')) {
                 currentPhase.status = 'success';
            } else if (message.includes('File uploaded successfully')) {
                 currentPhase.status = 'success';
            } else if (message.includes('Vectorization triggered successfully')) {
                 currentPhase.status = 'success';
            } else if (message.includes('Document is ready!')) {
                currentPhase.status = 'success';
            } else if (message.includes('Stream complete') && !message.includes('no slide data')) {
                currentPhase.status = 'success';
            }
        }
    });
    
    finalizeNetBuffer();

    return phases;
};

const HttpInfoBlock: React.FC<{ info: HttpInfo, type: 'Request' | 'Response' }> = ({ info, type }) => {
    const isSuccess = !info.status || (info.status >= 200 && info.status < 300);
    const Icon = type === 'Request' ? ArrowUpCircleIcon : ArrowDownCircleIcon;

    return (
        <div className="mt-2 text-xs font-mono">
            <div className="flex items-center gap-2">
                <Icon className={`w-4 h-4 ${isSuccess ? 'text-green-400' : 'text-red-400'}`} />
                <span className="font-bold">{type}:</span>
                <span className="font-semibold text-gray-300">{info.method}</span>
                {info.status && <span className={`font-bold px-1.5 py-0.5 rounded text-xs ${isSuccess ? 'bg-green-800 text-green-200' : 'bg-red-800 text-red-200'}`}>{info.status} {info.statusText}</span>}
                {info.duration && <span className="text-gray-400">({info.duration}ms)</span>}
            </div>
            <details className="pl-6 mt-1 text-gray-400">
                <summary className="cursor-pointer text-gray-500 hover:text-gray-300 text-[10px]">Details</summary>
                <div className="bg-gray-900 p-2 mt-1 rounded border border-gray-700">
                    <p className="whitespace-pre-wrap break-all"><strong>URL:</strong> {info.url}</p>
                    {info.headers && <pre className="whitespace-pre-wrap mt-1"><strong>Headers:</strong><br/>{info.headers}</pre>}
                    {info.body && <pre className="whitespace-pre-wrap mt-1"><strong>Body:</strong><br/>{info.body}</pre>}
                    {info.preview && <pre className="whitespace-pre-wrap mt-1"><strong>Preview:</strong><br/>{info.preview}</pre>}
                </div>
            </details>
        </div>
    );
};

const LogAnalysisPanel: React.FC<LogAnalysisPanelProps> = ({ logs }) => {
    const phases = useMemo(() => parseLogsForAnalysis(logs), [logs]);

    if (phases.length === 0) {
        return <div className="text-center text-gray-400">No document processing logs detected.</div>;
    }

    return (
        <div className="space-y-3 text-sm">
            {phases.map(phase => {
                const StatusIcon = phase.status === 'success' ? CheckCircleIcon : phase.status === 'error' ? XCircleIcon : ClockIcon;
                const color = phase.status === 'success' ? 'text-green-400' : phase.status === 'error' ? 'text-red-400' : 'text-yellow-400';

                return (
                    <div key={phase.key} className="bg-gray-900/50 p-3 rounded-lg border border-gray-700">
                        <h4 className={`font-bold flex items-center gap-2 ${color}`}>
                            <StatusIcon className="w-5 h-5" />
                            {phase.name}
                        </h4>
                        
                        {Object.entries(phase.data).map(([key, value]) => (
                            <div key={key} className="text-xs pl-7 text-gray-400">
                                <span className="font-semibold">{key}:</span> <span className="font-mono">{value}</span>
                            </div>
                        ))}

                        {phase.request && <HttpInfoBlock info={phase.request} type="Request" />}
                        {phase.response && <HttpInfoBlock info={phase.response} type="Response" />}

                        {phase.isPolling && phase.pollingAttempts && (
                             <details className="pl-6 mt-2 text-gray-400">
                                <summary className="cursor-pointer text-xs text-gray-500 hover:text-gray-300">{phase.pollingAttempts.length} Polling Attempts</summary>
                                <div className="space-y-2 mt-1">
                                    {phase.pollingAttempts.map((attempt, i) => <HttpInfoBlock key={i} info={attempt} type="Response" />)}
                                </div>
                            </details>
                        )}
                        
                        {phase.events.length > 0 && (
                            <div className="pl-6 mt-2 space-y-1">
                                {phase.events.map((event, i) => (
                                    <div key={i} className={`flex items-start gap-2 text-xs ${event.type === 'error' ? 'text-red-400' : 'text-gray-400'}`}>
                                        <InformationCircleIcon className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
                                        <span className="font-mono">{event.message}</span>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                );
            })}
        </div>
    );
};

export default LogAnalysisPanel;
