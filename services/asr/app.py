import os
import json
import logging
import tempfile
import subprocess
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
import numpy as np
from faster_whisper import WhisperModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/logs/asr.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
api = FastAPI(
    title="ASR API",
    description="Automatic Speech Recognition service with technical segment extraction",
    version="1.0.0"
)

# Models
class TranscriptionRequest(BaseModel):
    audio_url: Optional[str] = None
    audio_data: Optional[str] = None  # Base64 encoded
    language: Optional[str] = None
    extract_technical: bool = True
    segment_duration: int = 30  # seconds

class TranscriptionResponse(BaseModel):
    transcript: str
    segments: List[Dict[str, Any]]
    technical_segments: List[Dict[str, Any]]
    processing_time: float
    language: str
    audio_duration: float

class Segment(BaseModel):
    start: float
    end: float
    text: str
    confidence: float
    is_technical: bool = False
    technical_score: float = 0.0

class ASRService:
    """Main ASR service for speech recognition and processing"""
    
    def __init__(self):
        self.model = None
        self.model_size = os.getenv("ASR_MODEL", "medium")
        self.device = os.getenv("ASR_DEVICE", "cpu")
        
        # Technical keywords for segment classification
        self.technical_keywords = {
            "chemistry": [
                "molecule", "compound", "reaction", "catalyst", "pH", "concentration",
                "solvent", "polymer", "crystalline", "organic", "inorganic", "synthesis",
                "chemical", "formula", "element", "bond", "atomic", "molecular"
            ],
            "mechanical": [
                "force", "stress", "strain", "torque", "pressure", "tension", "compression",
                "beam", "shaft", "gear", "bearing", "joint", "mechanism", "machine",
                "newton", "pascal", "engineering", "structural", "material", "strength"
            ],
            "materials": [
                "steel", "aluminum", "composite", "ceramic", "polymer", "metal", "alloy",
                "hardness", "ductility", "brittleness", "elasticity", "plasticity",
                "microstructure", "grain", "phase", "crystal", "defect", "properties"
            ],
            "general_technical": [
                "analysis", "measurement", "calculation", "specification", "standard",
                "procedure", "method", "technique", "parameter", "variable", "coefficient",
                "equation", "algorithm", "optimization", "simulation", "modeling"
            ]
        }
        
        # Flatten all technical keywords
        self.all_technical_keywords = []
        for keywords in self.technical_keywords.values():
            self.all_technical_keywords.extend(keywords)
    
    async def initialize(self):
        """Initialize Whisper model"""
        try:
            logger.info(f"Loading Whisper model: {self.model_size} on {self.device}")
            
            # Initialize Faster Whisper model
            self.model = WhisperModel(
                self.model_size, 
                device=self.device,
                compute_type="float32" if self.device == "cpu" else "float16"
            )
            
            logger.info("Whisper model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Whisper model: {e}")
            raise
    
    def extract_audio_from_video(self, input_path: str, output_path: str) -> bool:
        """Extract audio from video using ffmpeg"""
        try:
            cmd = [
                "ffmpeg", "-i", input_path,
                "-ar", "16000",  # 16kHz sample rate for Whisper
                "-ac", "1",      # Mono
                "-c:a", "pcm_s16le",
                "-y",            # Overwrite output file
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"Audio extracted successfully: {output_path}")
                return True
            else:
                logger.error(f"FFmpeg error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Audio extraction timed out")
            return False
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            return False
    
    def get_audio_duration(self, file_path: str) -> float:
        """Get audio duration using ffprobe"""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-show_entries",
                "format=duration", "-of", "csv=p=0", file_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return float(result.stdout.strip())
            else:
                logger.warning(f"Could not get duration for {file_path}")
                return 0.0
                
        except Exception as e:
            logger.warning(f"Error getting audio duration: {e}")
            return 0.0
    
    def calculate_technical_score(self, text: str) -> float:
        """Calculate technical relevance score for text segment"""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        matches = 0
        
        for keyword in self.all_technical_keywords:
            if keyword in text_lower:
                matches += 1
        
        # Normalize by text length (rough estimate)
        words = len(text.split())
        if words == 0:
            return 0.0
        
        # Score is percentage of technical keywords found
        score = min(matches / max(words * 0.1, 1), 1.0)  # Cap at 1.0
        return score
    
    def classify_segments(self, segments: List[Dict], threshold: float = 0.15) -> List[Dict]:
        """Classify segments as technical or non-technical"""
        classified_segments = []
        
        for segment in segments:
            text = segment.get('text', '')
            technical_score = self.calculate_technical_score(text)
            is_technical = technical_score >= threshold
            
            classified_segment = {
                **segment,
                'is_technical': is_technical,
                'technical_score': technical_score
            }
            
            classified_segments.append(classified_segment)
        
        return classified_segments
    
    def merge_short_segments(self, segments: List[Dict], min_duration: float = 3.0) -> List[Dict]:
        """Merge very short segments with adjacent ones for better readability"""
        if not segments:
            return segments
        
        merged = []
        current_segment = segments[0].copy()
        
        for next_segment in segments[1:]:
            current_duration = current_segment['end'] - current_segment['start']
            
            # If current segment is too short, merge with next
            if current_duration < min_duration:
                current_segment['end'] = next_segment['end']
                current_segment['text'] += ' ' + next_segment['text']
                current_segment['confidence'] = (
                    current_segment['confidence'] + next_segment['confidence']
                ) / 2
                # Recalculate technical score for merged text
                technical_score = self.calculate_technical_score(current_segment['text'])
                current_segment['technical_score'] = technical_score
                current_segment['is_technical'] = technical_score >= 0.15
            else:
                merged.append(current_segment)
                current_segment = next_segment.copy()
        
        # Don't forget the last segment
        merged.append(current_segment)
        
        return merged
    
    async def transcribe_audio(
        self, 
        file_path: str, 
        language: Optional[str] = None,
        extract_technical: bool = True
    ) -> Dict[str, Any]:
        """Transcribe audio file and extract technical segments"""
        start_time = datetime.utcnow()
        
        try:
            # Get audio duration
            audio_duration = self.get_audio_duration(file_path)
            
            # Transcribe with Whisper
            logger.info(f"Starting transcription of {file_path}")
            
            segments, info = self.model.transcribe(
                file_path,
                language=language,
                beam_size=5,
                word_timestamps=True
            )
            
            # Convert segments to list and add confidence scores
            segment_list = []
            full_transcript_parts = []
            
            for segment in segments:
                segment_dict = {
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text.strip(),
                    'confidence': segment.avg_logprob if hasattr(segment, 'avg_logprob') else 0.0,
                    'words': []
                }
                
                # Add word-level timestamps if available
                if hasattr(segment, 'words') and segment.words:
                    for word in segment.words:
                        word_dict = {
                            'word': word.word,
                            'start': word.start,
                            'end': word.end,
                            'probability': word.probability if hasattr(word, 'probability') else 0.0
                        }
                        segment_dict['words'].append(word_dict)
                
                segment_list.append(segment_dict)
                full_transcript_parts.append(segment.text.strip())
            
            # Create full transcript
            full_transcript = ' '.join(full_transcript_parts)
            
            # Classify technical segments if requested
            if extract_technical:
                segment_list = self.classify_segments(segment_list)
                segment_list = self.merge_short_segments(segment_list)
            
            # Filter technical segments
            technical_segments = [
                seg for seg in segment_list 
                if seg.get('is_technical', False)
            ] if extract_technical else []
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            logger.info(
                f"Transcription completed in {processing_time:.2f}s. "
                f"Generated {len(segment_list)} segments, "
                f"{len(technical_segments)} technical segments."
            )
            
            return {
                'transcript': full_transcript,
                'segments': segment_list,
                'technical_segments': technical_segments,
                'processing_time': processing_time,
                'language': info.language if hasattr(info, 'language') else language or 'unknown',
                'audio_duration': audio_duration
            }
            
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise

# Global service instance
asr_service = ASRService()

@api.on_event("startup")
async def startup_event():
    """Initialize ASR service on startup"""
    await asr_service.initialize()

@api.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": asr_service.model is not None,
        "model_size": asr_service.model_size,
        "device": asr_service.device
    }

@api.post("/transcribe/file", response_model=TranscriptionResponse)
async def transcribe_file(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    extract_technical: bool = Form(True)
):
    """Transcribe an uploaded audio/video file"""
    if not asr_service.model:
        raise HTTPException(status_code=503, detail="ASR model not loaded")
    
    try:
        # Save uploaded file temporarily
        file_extension = Path(file.filename).suffix.lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # If it's a video file, extract audio first
            audio_file_path = temp_file_path
            if file_extension in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                audio_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                audio_file_path = audio_temp_file.name
                audio_temp_file.close()
                
                success = asr_service.extract_audio_from_video(temp_file_path, audio_file_path)
                if not success:
                    raise HTTPException(status_code=400, detail="Failed to extract audio from video")
            
            # Transcribe the audio
            result = await asr_service.transcribe_audio(
                audio_file_path, language, extract_technical
            )
            
            return TranscriptionResponse(**result)
            
        finally:
            # Clean up temporary files
            try:
                os.unlink(temp_file_path)
                if audio_file_path != temp_file_path:
                    os.unlink(audio_file_path)
            except:
                pass
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@api.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_request(request: TranscriptionRequest):
    """Transcribe audio from URL or base64 data"""
    if not asr_service.model:
        raise HTTPException(status_code=503, detail="ASR model not loaded")
    
    if not request.audio_url and not request.audio_data:
        raise HTTPException(status_code=400, detail="Either audio_url or audio_data must be provided")
    
    try:
        # For now, return a placeholder for URL/base64 processing
        # This would require additional implementation for URL downloading and base64 decoding
        if request.audio_url:
            raise HTTPException(status_code=501, detail="URL transcription not yet implemented")
        
        if request.audio_data:
            raise HTTPException(status_code=501, detail="Base64 audio data transcription not yet implemented")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription request error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@api.get("/technical/keywords")
async def get_technical_keywords():
    """Get technical keywords used for segment classification"""
    return {
        "categories": asr_service.technical_keywords,
        "total_keywords": len(asr_service.all_technical_keywords)
    }

@api.post("/technical/analyze")
async def analyze_text_technical(text: str):
    """Analyze text for technical content (debugging endpoint)"""
    score = asr_service.calculate_technical_score(text)
    is_technical = score >= 0.15
    
    # Find matching keywords
    text_lower = text.lower()
    matches = []
    for keyword in asr_service.all_technical_keywords:
        if keyword in text_lower:
            matches.append(keyword)
    
    return {
        "text": text,
        "technical_score": score,
        "is_technical": is_technical,
        "matching_keywords": matches,
        "word_count": len(text.split())
    }

@api.get("/models/info")
async def get_model_info():
    """Get information about the loaded ASR model"""
    return {
        "model_size": asr_service.model_size,
        "device": asr_service.device,
        "model_loaded": asr_service.model is not None,
        "supported_languages": [
            "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", 
            "ar", "hi", "tr", "pl", "nl", "sv", "da", "no", "fi"
        ],
        "supported_formats": [
            "wav", "mp3", "m4a", "flac", "ogg",  # Audio
            "mp4", "avi", "mov", "mkv", "webm"   # Video (audio extraction)
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8000)