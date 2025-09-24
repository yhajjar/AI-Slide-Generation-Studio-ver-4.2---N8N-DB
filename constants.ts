import { GeneralContentType, MicrolearningContentType, InteractiveElement } from './types';

export const MIN_SLIDES_GENERAL = 1;
export const MAX_SLIDES_GENERAL = 10;
export const MAX_SLIDES_MICRO = 10;
export const MIN_SLIDES_MICRO = 1;

export const ALL_GENERAL_CONTENT_TYPES: GeneralContentType[] = [
  GeneralContentType.OVERVIEW,
  GeneralContentType.OBJECTIVES,
  GeneralContentType.CONTENT,
  GeneralContentType.SUMMARY,
];

export const ALL_MICROLEARNING_CONTENT_TYPES: MicrolearningContentType[] = [
  MicrolearningContentType.OBJECTIVE,
  MicrolearningContentType.CONTENT,
  MicrolearningContentType.SUMMARY,
];

export const ALL_INTERACTIVE_ELEMENTS: InteractiveElement[] = [
  InteractiveElement.FLASHCARDS,
  InteractiveElement.INFOGRAPHICS,
  InteractiveElement.QUIZ,
  InteractiveElement.MATCHING,
];