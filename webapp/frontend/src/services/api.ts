import { API_BASE_URL, MIN_CHAR_COUNT, MAX_CHAR_COUNT } from './config';

export interface PredictionRequest {
  text1: string;
  text2: string;
}

export interface PredictionResponse {
  same_author_probability: number;
}

export class ApiError extends Error {
  statusCode?: number;
  details?: unknown;

  constructor(
    message: string,
    statusCode?: number,
    details?: unknown
  ) {
    super(message);
    this.name = 'ApiError';
    this.statusCode = statusCode;
    this.details = details;
  }
}

export async function predictAuthorship(
  text1: string,
  text2: string
): Promise<PredictionResponse> {
  try {
    var text1_length = text1.trim().length;
    var text2_length = text2.trim().length;

    if (text1_length < MIN_CHAR_COUNT ||  text2_length < MIN_CHAR_COUNT) {
      throw new ApiError(`Text has less then ${MIN_CHAR_COUNT} characters. Please provide longer text.`);
    } else if (text1_length >  MAX_CHAR_COUNT || text2_length > MAX_CHAR_COUNT) {
      throw new ApiError(`Text has more then ${MAX_CHAR_COUNT} characters. Please provide shorter text.`);
    }

    const response = await fetch(`${API_BASE_URL}/predict/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      //convert to JSON
      body: JSON.stringify({
        text1,
        text2,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new ApiError(
        errorData.detail || `API request failed with status ${response.status}`,
        response.status,
        errorData
      );
    }

    const data: PredictionResponse = await response.json();
    return data;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }

    // Network or other errors
    throw new ApiError(
      'Failed to connect to the API. Please ensure the backend is running.',
      undefined,
      error
    );
  }
}

export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.ok;
  } catch {
    return false;
  }
}