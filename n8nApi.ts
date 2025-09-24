// n8nApi.ts
import { CourseType } from './types';

const N8N_UPSERT_URL = 'https://n8n.myapps.mylabs.click/webhook/upsert';
const N8N_RETRIEVE_URL = 'https://n8n.myapps.mylabs.click/webhook/retrieve';

const log = (onLog?: (s: string) => void, msg = "") => {
  if (!onLog) return;
  onLog(`[${new Date().toISOString()}] [n8nAPI] ${msg}`);
};

interface UpsertResponse {
  sourceId: string;
}

export async function upsertDocument(
  file: File,
  metadata: { runId: string; topic: string },
  onLog?: (message: string) => void
): Promise<UpsertResponse> {
  log(onLog, `Uploading document "${file.name}" for topic "${metadata.topic}" with runId ${metadata.runId}`);
  
  const formData = new FormData();
  // Explicitly append fields with the correct keys as required by the n8n webhook.
  formData.append('file', file);
  formData.append('runId', metadata.runId);
  formData.append('topic', metadata.topic);

  // Add logging to verify the keys being sent in FormData, to help debug the discrepancy.
  if (onLog) {
      const keys = Array.from(formData.keys()).join(', ');
      log(onLog, `FormData keys being sent: [${keys}]`);
  }

  try {
    const response = await fetch(N8N_UPSERT_URL, {
      method: 'POST',
      body: formData,
    });

    log(onLog, `Upload response status: ${response.status}`);
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`n8n document upsert failed: ${response.status} - ${errorText}`);
    }

    const result = await response.json();
    log(onLog, `Upload response JSON: ${JSON.stringify(result)}`);
    
    // The n8n /upsert endpoint must return a `sourceId` to identify the document for retrieval.
    if (!result.sourceId) {
        throw new Error('`sourceId` not found in n8n upsert response.');
    }

    return result as UpsertResponse;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    log(onLog, `Upsert error: ${message}`);
    throw error;
  }
}

export interface SlideRequestInfo {
    id: number;
    contentType: string;
    autoMode: boolean;
    userContent: string;
    interactive: string;
}

interface RetrieveParams {
  runId: string;
  sourceId: string;
  topic: string;
  courseType: CourseType;
  slides: SlideRequestInfo[];
}

interface RetrieveResponseItem {
    id: number;
    SlideContent: string;
}


export async function retrieveGroundTruth(
  params: RetrieveParams,
  onLog?: (message: string) => void
): Promise<RetrieveResponseItem[]> {
  log(onLog, `Retrieving ground truth for sourceId "${params.sourceId}" with runId ${params.runId}`);

  try {
    const response = await fetch(N8N_RETRIEVE_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(params),
    });
    
    log(onLog, `Retrieve response status: ${response.status}`);

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`n8n ground truth retrieval failed: ${response.status} - ${errorText}`);
    }

    const result = await response.json();
    log(onLog, `Retrieve response JSON: ${JSON.stringify(result)}`);
    
    if (!result || !Array.isArray(result)) {
        throw new Error('n8n retrieve response is not a valid array.');
    }

    // Validate that each item has an ID to ensure correct slide mapping.
    for (const item of result) {
        if (typeof item.id !== 'number') {
            throw new Error(`n8n retrieve response item is missing a numeric 'id' field. Item: ${JSON.stringify(item)}`);
        }
    }

    return result as RetrieveResponseItem[];

  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    log(onLog, `Retrieve error: ${message}`);
    throw error;
  }
}