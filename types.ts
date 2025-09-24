export enum CourseType {
  GENERAL = 'general',
  MICROLEARNING = 'microlearning',
}

export enum StructureMethod {
  AI = 'ai',
  USER = 'user',
  DOCUMENT = 'document',
}

export enum GeneralContentType {
  OVERVIEW = 'Overview',
  OBJECTIVES = 'Objectives',
  CONTENT = 'Content',
  SUMMARY = 'Summary',
}

export enum MicrolearningContentType {
  OBJECTIVE = 'Objective',
  CONTENT = 'Content',
  SUMMARY = 'Summary',
}

export enum InteractiveElement {
  FLASHCARDS = 'Flashcards',
  INFOGRAPHICS = 'Infographics',
  QUIZ = 'Quiz',
  MATCHING = 'Matching',
}

export type AgenticMode = "free" | "strict";

export type KbStatus =
  | 'idle'
  | 'uploading'
  | 'registering'
  | 'vectorizing'
  | 'polling'
  | 'ready'
  | 'error';


export interface GeneralCourseSlide {
  id: number;
  userContent: string;
  contentType: GeneralContentType;
  autoMode: boolean;
  generatedContent?: string;
}

export interface GeneralCourseData {
  courseType: CourseType.GENERAL;
  courseTopic: string;
  structureMethod: StructureMethod;
  selectedContentTypes: GeneralContentType[];
  slideCount: number;
  slides: GeneralCourseSlide[];
  kbId?: string;
  fileName?: string;
}

export interface MicrolearningSlide {
  id: number;
  autoMode: boolean;
  userContent: string;
  contentType: MicrolearningContentType;
  interactives: InteractiveElement[];
  generatedContent?: string;
  generatedInteractives?: any;
}

export interface MicrolearningCourseData {
  courseType: CourseType.MICROLEARNING;
  courseTopic: string;
  structureMethod: StructureMethod;
  slideCount: number;
  selectedContentTypes: MicrolearningContentType[];
  slides: MicrolearningSlide[];
  // FIX: Add optional 'selectedInteractives' to support selection of interactive elements for microlearning courses.
  selectedInteractives?: InteractiveElement[];
  kbId?: string;
  fileName?: string;
}

export type CourseData = GeneralCourseData | MicrolearningCourseData;

export interface SlideGenState {
  step: number;
  courseData: Partial<CourseData>;
  kbStatus: KbStatus;
  kbError: string | null;
  mode: AgenticMode;
  isLoading: boolean;
  isRetrievingContent: boolean;
  isExporting: boolean;
  generatedSlides: GeneratedSlide[];
  runId: string | null; // client-side correlation id for n8n
  glmConversationId: string | null; // For GLM revisions
  error: string | null;
  lastPrompt: string;
  apiLogs: string[];
}

export interface GeneratedSlide {
  html: string;
  pageNumber: number;
  draft?: string;
  complete?: boolean;
}