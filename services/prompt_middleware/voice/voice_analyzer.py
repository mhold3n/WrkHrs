#!/usr/bin/env python3
"""
Advanced Voice Analysis for Subtext Inference
Analyzes voice patterns to extract implied meaning and emotional context
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Any, Tuple, Optional
import logging
from dataclasses import dataclass
from scipy import signal
from scipy.stats import skew, kurtosis
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

@dataclass
class VoiceFeatures:
    """Comprehensive voice feature set"""
    # Prosodic features
    pitch_mean: float
    pitch_std: float
    pitch_range: float
    pitch_skew: float
    pitch_kurtosis: float
    
    # Rhythm features
    tempo: float
    rhythm_regularity: float
    pause_frequency: float
    pause_duration_mean: float
    
    # Energy features
    energy_mean: float
    energy_std: float
    energy_skew: float
    energy_kurtosis: float
    
    # Spectral features
    spectral_centroid_mean: float
    spectral_centroid_std: float
    spectral_rolloff_mean: float
    spectral_rolloff_std: float
    spectral_bandwidth_mean: float
    spectral_bandwidth_std: float
    zero_crossing_rate_mean: float
    zero_crossing_rate_std: float
    
    # MFCC features
    mfcc_mean: np.ndarray
    mfcc_std: np.ndarray
    
    # Formant features
    f1_mean: float
    f2_mean: float
    f3_mean: float
    f1_std: float
    f2_std: float
    f3_std: float
    
    # Voice quality features
    jitter: float
    shimmer: float
    hnr: float  # Harmonics-to-noise ratio
    
    # Emotional features
    valence: float
    arousal: float
    dominance: float

@dataclass
class SubtextInference:
    """Inferred subtext from voice analysis"""
    emotional_state: Dict[str, float]
    confidence_level: float
    intentional_ambiguity: float
    urgency_level: float
    social_dominance: float
    deception_indicators: float
    cognitive_load: float
    cultural_markers: Dict[str, float]
    personality_traits: Dict[str, float]

class AdvancedVoiceAnalyzer:
    """Advanced voice analysis for subtext inference"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.n_mfcc = 13
        self.n_fft = 2048
        self.hop_length = 512
        self.scaler = StandardScaler()
        
    def extract_comprehensive_features(self, audio_data: np.ndarray) -> VoiceFeatures:
        """Extract comprehensive voice features"""
        try:
            # Ensure audio is mono
            if len(audio_data.shape) > 1:
                audio_data = librosa.to_mono(audio_data)
            
            # Resample if necessary
            if len(audio_data) > 0:
                audio_data = librosa.resample(audio_data, orig_sr=len(audio_data), target_sr=self.sample_rate)
            
            # Extract prosodic features
            prosodic_features = self._extract_prosodic_features(audio_data)
            
            # Extract rhythm features
            rhythm_features = self._extract_rhythm_features(audio_data)
            
            # Extract energy features
            energy_features = self._extract_energy_features(audio_data)
            
            # Extract spectral features
            spectral_features = self._extract_spectral_features(audio_data)
            
            # Extract MFCC features
            mfcc_features = self._extract_mfcc_features(audio_data)
            
            # Extract formant features
            formant_features = self._extract_formant_features(audio_data)
            
            # Extract voice quality features
            voice_quality_features = self._extract_voice_quality_features(audio_data)
            
            # Extract emotional features
            emotional_features = self._extract_emotional_features(audio_data)
            
            return VoiceFeatures(
                # Prosodic
                pitch_mean=prosodic_features['pitch_mean'],
                pitch_std=prosodic_features['pitch_std'],
                pitch_range=prosodic_features['pitch_range'],
                pitch_skew=prosodic_features['pitch_skew'],
                pitch_kurtosis=prosodic_features['pitch_kurtosis'],
                
                # Rhythm
                tempo=rhythm_features['tempo'],
                rhythm_regularity=rhythm_features['rhythm_regularity'],
                pause_frequency=rhythm_features['pause_frequency'],
                pause_duration_mean=rhythm_features['pause_duration_mean'],
                
                # Energy
                energy_mean=energy_features['energy_mean'],
                energy_std=energy_features['energy_std'],
                energy_skew=energy_features['energy_skew'],
                energy_kurtosis=energy_features['energy_kurtosis'],
                
                # Spectral
                spectral_centroid_mean=spectral_features['spectral_centroid_mean'],
                spectral_centroid_std=spectral_features['spectral_centroid_std'],
                spectral_rolloff_mean=spectral_features['spectral_rolloff_mean'],
                spectral_rolloff_std=spectral_features['spectral_rolloff_std'],
                spectral_bandwidth_mean=spectral_features['spectral_bandwidth_mean'],
                spectral_bandwidth_std=spectral_features['spectral_bandwidth_std'],
                zero_crossing_rate_mean=spectral_features['zero_crossing_rate_mean'],
                zero_crossing_rate_std=spectral_features['zero_crossing_rate_std'],
                
                # MFCC
                mfcc_mean=mfcc_features['mfcc_mean'],
                mfcc_std=mfcc_features['mfcc_std'],
                
                # Formants
                f1_mean=formant_features['f1_mean'],
                f2_mean=formant_features['f2_mean'],
                f3_mean=formant_features['f3_mean'],
                f1_std=formant_features['f1_std'],
                f2_std=formant_features['f2_std'],
                f3_std=formant_features['f3_std'],
                
                # Voice quality
                jitter=voice_quality_features['jitter'],
                shimmer=voice_quality_features['shimmer'],
                hnr=voice_quality_features['hnr'],
                
                # Emotional
                valence=emotional_features['valence'],
                arousal=emotional_features['arousal'],
                dominance=emotional_features['dominance']
            )
            
        except Exception as e:
            logger.error(f"Error extracting voice features: {e}")
            # Return default features
            return self._get_default_features()
    
    def _extract_prosodic_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Extract prosodic features (pitch, intonation)"""
        try:
            # Fundamental frequency using YIN algorithm
            f0 = librosa.yin(audio_data, fmin=50, fmax=400, sr=self.sample_rate)
            f0_clean = f0[~np.isnan(f0)]
            
            if len(f0_clean) == 0:
                return {
                    'pitch_mean': 0.0,
                    'pitch_std': 0.0,
                    'pitch_range': 0.0,
                    'pitch_skew': 0.0,
                    'pitch_kurtosis': 0.0
                }
            
            return {
                'pitch_mean': float(np.mean(f0_clean)),
                'pitch_std': float(np.std(f0_clean)),
                'pitch_range': float(np.max(f0_clean) - np.min(f0_clean)),
                'pitch_skew': float(skew(f0_clean)),
                'pitch_kurtosis': float(kurtosis(f0_clean))
            }
        except Exception as e:
            logger.error(f"Error extracting prosodic features: {e}")
            return {'pitch_mean': 0.0, 'pitch_std': 0.0, 'pitch_range': 0.0, 'pitch_skew': 0.0, 'pitch_kurtosis': 0.0}
    
    def _extract_rhythm_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Extract rhythm and timing features"""
        try:
            # Tempo estimation
            tempo, beats = librosa.beat.beat_track(y=audio_data, sr=self.sample_rate)
            
            # Rhythm regularity (coefficient of variation of inter-beat intervals)
            if len(beats) > 1:
                inter_beat_intervals = np.diff(beats)
                rhythm_regularity = 1.0 / (1.0 + np.std(inter_beat_intervals) / np.mean(inter_beat_intervals))
            else:
                rhythm_regularity = 0.0
            
            # Pause detection
            rms = librosa.feature.rms(y=audio_data)[0]
            pause_threshold = np.mean(rms) * 0.1
            pauses = rms < pause_threshold
            
            # Pause frequency and duration
            pause_changes = np.diff(pauses.astype(int))
            pause_starts = np.where(pause_changes == 1)[0]
            pause_ends = np.where(pause_changes == -1)[0]
            
            if len(pause_starts) > 0 and len(pause_ends) > 0:
                pause_durations = pause_ends - pause_starts[:len(pause_ends)]
                pause_frequency = len(pause_starts) / (len(audio_data) / self.sample_rate)
                pause_duration_mean = np.mean(pause_durations) / self.sample_rate
            else:
                pause_frequency = 0.0
                pause_duration_mean = 0.0
            
            return {
                'tempo': float(tempo),
                'rhythm_regularity': float(rhythm_regularity),
                'pause_frequency': float(pause_frequency),
                'pause_duration_mean': float(pause_duration_mean)
            }
        except Exception as e:
            logger.error(f"Error extracting rhythm features: {e}")
            return {'tempo': 0.0, 'rhythm_regularity': 0.0, 'pause_frequency': 0.0, 'pause_duration_mean': 0.0}
    
    def _extract_energy_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Extract energy and intensity features"""
        try:
            # RMS energy
            rms = librosa.feature.rms(y=audio_data)[0]
            
            return {
                'energy_mean': float(np.mean(rms)),
                'energy_std': float(np.std(rms)),
                'energy_skew': float(skew(rms)),
                'energy_kurtosis': float(kurtosis(rms))
            }
        except Exception as e:
            logger.error(f"Error extracting energy features: {e}")
            return {'energy_mean': 0.0, 'energy_std': 0.0, 'energy_skew': 0.0, 'energy_kurtosis': 0.0}
    
    def _extract_spectral_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Extract spectral features"""
        try:
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)[0]
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=self.sample_rate)[0]
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            
            return {
                'spectral_centroid_mean': float(np.mean(spectral_centroids)),
                'spectral_centroid_std': float(np.std(spectral_centroids)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'spectral_rolloff_std': float(np.std(spectral_rolloff)),
                'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
                'spectral_bandwidth_std': float(np.std(spectral_bandwidth)),
                'zero_crossing_rate_mean': float(np.mean(zcr)),
                'zero_crossing_rate_std': float(np.std(zcr))
            }
        except Exception as e:
            logger.error(f"Error extracting spectral features: {e}")
            return {
                'spectral_centroid_mean': 0.0, 'spectral_centroid_std': 0.0,
                'spectral_rolloff_mean': 0.0, 'spectral_rolloff_std': 0.0,
                'spectral_bandwidth_mean': 0.0, 'spectral_bandwidth_std': 0.0,
                'zero_crossing_rate_mean': 0.0, 'zero_crossing_rate_std': 0.0
            }
    
    def _extract_mfcc_features(self, audio_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract MFCC features"""
        try:
            mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=self.n_mfcc)
            
            return {
                'mfcc_mean': np.mean(mfccs, axis=1),
                'mfcc_std': np.std(mfccs, axis=1)
            }
        except Exception as e:
            logger.error(f"Error extracting MFCC features: {e}")
            return {
                'mfcc_mean': np.zeros(self.n_mfcc),
                'mfcc_std': np.zeros(self.n_mfcc)
            }
    
    def _extract_formant_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Extract formant features (simplified)"""
        try:
            # Simplified formant extraction using spectral peaks
            stft = librosa.stft(audio_data)
            magnitude = np.abs(stft)
            
            # Find spectral peaks (simplified formant detection)
            freqs = librosa.fft_frequencies(sr=self.sample_rate)
            
            # Find peaks in the magnitude spectrum
            peaks, _ = signal.find_peaks(np.mean(magnitude, axis=1), height=np.max(np.mean(magnitude, axis=1)) * 0.1)
            
            if len(peaks) >= 3:
                f1 = freqs[peaks[0]]
                f2 = freqs[peaks[1]]
                f3 = freqs[peaks[2]]
                f1_std = np.std(freqs[peaks[0]])
                f2_std = np.std(freqs[peaks[1]])
                f3_std = np.std(freqs[peaks[2]])
            else:
                f1 = f2 = f3 = 0.0
                f1_std = f2_std = f3_std = 0.0
            
            return {
                'f1_mean': float(f1),
                'f2_mean': float(f2),
                'f3_mean': float(f3),
                'f1_std': float(f1_std),
                'f2_std': float(f2_std),
                'f3_std': float(f3_std)
            }
        except Exception as e:
            logger.error(f"Error extracting formant features: {e}")
            return {
                'f1_mean': 0.0, 'f2_mean': 0.0, 'f3_mean': 0.0,
                'f1_std': 0.0, 'f2_std': 0.0, 'f3_std': 0.0
            }
    
    def _extract_voice_quality_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Extract voice quality features (jitter, shimmer, HNR)"""
        try:
            # Simplified jitter and shimmer calculation
            f0 = librosa.yin(audio_data, fmin=50, fmax=400, sr=self.sample_rate)
            f0_clean = f0[~np.isnan(f0)]
            
            if len(f0_clean) < 2:
                return {'jitter': 0.0, 'shimmer': 0.0, 'hnr': 0.0}
            
            # Jitter (pitch period variability)
            jitter = np.std(np.diff(f0_clean)) / np.mean(f0_clean)
            
            # Shimmer (amplitude variability) - simplified
            rms = librosa.feature.rms(y=audio_data)[0]
            shimmer = np.std(rms) / np.mean(rms)
            
            # Harmonics-to-noise ratio (simplified)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
            hnr = np.mean(spectral_centroids) / (np.std(spectral_centroids) + 1e-10)
            
            return {
                'jitter': float(jitter),
                'shimmer': float(shimmer),
                'hnr': float(hnr)
            }
        except Exception as e:
            logger.error(f"Error extracting voice quality features: {e}")
            return {'jitter': 0.0, 'shimmer': 0.0, 'hnr': 0.0}
    
    def _extract_emotional_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Extract emotional features (valence, arousal, dominance)"""
        try:
            # Simplified emotional feature extraction
            # In practice, this would use a trained emotion recognition model
            
            # Valence (positive/negative emotion)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
            valence = np.tanh((np.mean(spectral_centroids) - 2000) / 1000)  # Normalize to [-1, 1]
            
            # Arousal (activation level)
            rms = librosa.feature.rms(y=audio_data)[0]
            arousal = np.tanh(np.mean(rms) * 10)  # Normalize to [-1, 1]
            
            # Dominance (control/power)
            f0 = librosa.yin(audio_data, fmin=50, fmax=400, sr=self.sample_rate)
            f0_clean = f0[~np.isnan(f0)]
            if len(f0_clean) > 0:
                dominance = np.tanh((np.mean(f0_clean) - 200) / 100)  # Normalize to [-1, 1]
            else:
                dominance = 0.0
            
            return {
                'valence': float(valence),
                'arousal': float(arousal),
                'dominance': float(dominance)
            }
        except Exception as e:
            logger.error(f"Error extracting emotional features: {e}")
            return {'valence': 0.0, 'arousal': 0.0, 'dominance': 0.0}
    
    def _get_default_features(self) -> VoiceFeatures:
        """Return default features when extraction fails"""
        return VoiceFeatures(
            pitch_mean=0.0, pitch_std=0.0, pitch_range=0.0, pitch_skew=0.0, pitch_kurtosis=0.0,
            tempo=0.0, rhythm_regularity=0.0, pause_frequency=0.0, pause_duration_mean=0.0,
            energy_mean=0.0, energy_std=0.0, energy_skew=0.0, energy_kurtosis=0.0,
            spectral_centroid_mean=0.0, spectral_centroid_std=0.0,
            spectral_rolloff_mean=0.0, spectral_rolloff_std=0.0,
            spectral_bandwidth_mean=0.0, spectral_bandwidth_std=0.0,
            zero_crossing_rate_mean=0.0, zero_crossing_rate_std=0.0,
            mfcc_mean=np.zeros(self.n_mfcc), mfcc_std=np.zeros(self.n_mfcc),
            f1_mean=0.0, f2_mean=0.0, f3_mean=0.0,
            f1_std=0.0, f2_std=0.0, f3_std=0.0,
            jitter=0.0, shimmer=0.0, hnr=0.0,
            valence=0.0, arousal=0.0, dominance=0.0
        )
    
    def infer_subtext(self, features: VoiceFeatures) -> SubtextInference:
        """Infer subtext from voice features"""
        
        # Emotional state inference
        emotional_state = self._infer_emotional_state(features)
        
        # Confidence level
        confidence_level = self._calculate_confidence(features)
        
        # Intentional ambiguity
        intentional_ambiguity = self._detect_intentional_ambiguity(features)
        
        # Urgency level
        urgency_level = self._assess_urgency(features)
        
        # Social dominance
        social_dominance = self._assess_social_dominance(features)
        
        # Deception indicators
        deception_indicators = self._detect_deception_indicators(features)
        
        # Cognitive load
        cognitive_load = self._assess_cognitive_load(features)
        
        # Cultural markers
        cultural_markers = self._identify_cultural_markers(features)
        
        # Personality traits
        personality_traits = self._infer_personality_traits(features)
        
        return SubtextInference(
            emotional_state=emotional_state,
            confidence_level=confidence_level,
            intentional_ambiguity=intentional_ambiguity,
            urgency_level=urgency_level,
            social_dominance=social_dominance,
            deception_indicators=deception_indicators,
            cognitive_load=cognitive_load,
            cultural_markers=cultural_markers,
            personality_traits=personality_traits
        )
    
    def _infer_emotional_state(self, features: VoiceFeatures) -> Dict[str, float]:
        """Infer emotional state from voice features"""
        emotions = {
            'anger': 0.0,
            'joy': 0.0,
            'sadness': 0.0,
            'fear': 0.0,
            'surprise': 0.0,
            'disgust': 0.0,
            'neutral': 0.0
        }
        
        # Anger: High pitch, high energy, high arousal
        if features.pitch_mean > 200 and features.energy_mean > 0.1 and features.arousal > 0.5:
            emotions['anger'] += 0.4
        
        # Joy: High pitch, high energy, positive valence
        if features.pitch_mean > 180 and features.energy_mean > 0.08 and features.valence > 0.3:
            emotions['joy'] += 0.4
        
        # Sadness: Low pitch, low energy, negative valence
        if features.pitch_mean < 150 and features.energy_mean < 0.05 and features.valence < -0.3:
            emotions['sadness'] += 0.4
        
        # Fear: High pitch variance, high arousal, negative valence
        if features.pitch_std > 50 and features.arousal > 0.3 and features.valence < -0.2:
            emotions['fear'] += 0.4
        
        # Surprise: High pitch, high arousal
        if features.pitch_mean > 200 and features.arousal > 0.6:
            emotions['surprise'] += 0.3
        
        # Disgust: Low pitch, negative valence
        if features.pitch_mean < 140 and features.valence < -0.4:
            emotions['disgust'] += 0.3
        
        # Normalize
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: v/total for k, v in emotions.items()}
        else:
            emotions['neutral'] = 1.0
        
        return emotions
    
    def _calculate_confidence(self, features: VoiceFeatures) -> float:
        """Calculate confidence level in the analysis"""
        # Higher confidence with more stable features
        confidence = 0.5
        
        # Lower pitch variance = higher confidence
        if features.pitch_std < 30:
            confidence += 0.2
        
        # Higher rhythm regularity = higher confidence
        confidence += features.rhythm_regularity * 0.2
        
        # Lower jitter and shimmer = higher confidence
        if features.jitter < 0.05:
            confidence += 0.1
        if features.shimmer < 0.1:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _detect_intentional_ambiguity(self, features: VoiceFeatures) -> float:
        """Detect intentional ambiguity in speech"""
        ambiguity = 0.0
        
        # High pitch variance suggests uncertainty
        if features.pitch_std > 60:
            ambiguity += 0.3
        
        # Irregular rhythm suggests hesitation
        if features.rhythm_regularity < 0.5:
            ambiguity += 0.2
        
        # Frequent pauses suggest thinking/hesitation
        if features.pause_frequency > 2.0:
            ambiguity += 0.2
        
        # High cognitive load suggests complexity/ambiguity
        if features.jitter > 0.1 or features.shimmer > 0.15:
            ambiguity += 0.3
        
        return min(ambiguity, 1.0)
    
    def _assess_urgency(self, features: VoiceFeatures) -> float:
        """Assess urgency level from voice features"""
        urgency = 0.0
        
        # High tempo suggests urgency
        if features.tempo > 120:
            urgency += 0.3
        
        # High energy suggests urgency
        if features.energy_mean > 0.1:
            urgency += 0.2
        
        # High arousal suggests urgency
        if features.arousal > 0.5:
            urgency += 0.2
        
        # High pitch suggests urgency
        if features.pitch_mean > 200:
            urgency += 0.2
        
        # Few pauses suggests urgency
        if features.pause_frequency < 1.0:
            urgency += 0.1
        
        return min(urgency, 1.0)
    
    def _assess_social_dominance(self, features: VoiceFeatures) -> float:
        """Assess social dominance from voice features"""
        dominance = 0.0
        
        # Higher pitch suggests dominance (in some contexts)
        if features.pitch_mean > 180:
            dominance += 0.2
        
        # Higher energy suggests dominance
        if features.energy_mean > 0.08:
            dominance += 0.2
        
        # Lower jitter suggests confidence/dominance
        if features.jitter < 0.05:
            dominance += 0.2
        
        # Regular rhythm suggests confidence
        if features.rhythm_regularity > 0.7:
            dominance += 0.2
        
        # Higher dominance score from features
        if features.dominance > 0.3:
            dominance += 0.2
        
        return min(dominance, 1.0)
    
    def _detect_deception_indicators(self, features: VoiceFeatures) -> float:
        """Detect potential deception indicators"""
        deception = 0.0
        
        # High jitter suggests stress/deception
        if features.jitter > 0.08:
            deception += 0.3
        
        # High shimmer suggests stress/deception
        if features.shimmer > 0.12:
            deception += 0.2
        
        # Irregular rhythm suggests stress
        if features.rhythm_regularity < 0.4:
            deception += 0.2
        
        # High cognitive load suggests deception
        if features.pause_frequency > 3.0:
            deception += 0.2
        
        # Low HNR suggests stress
        if features.hnr < 10:
            deception += 0.1
        
        return min(deception, 1.0)
    
    def _assess_cognitive_load(self, features: VoiceFeatures) -> float:
        """Assess cognitive load from voice features"""
        cognitive_load = 0.0
        
        # High pause frequency suggests thinking
        if features.pause_frequency > 2.0:
            cognitive_load += 0.3
        
        # Irregular rhythm suggests cognitive effort
        if features.rhythm_regularity < 0.5:
            cognitive_load += 0.2
        
        # High jitter suggests cognitive stress
        if features.jitter > 0.06:
            cognitive_load += 0.2
        
        # Lower energy suggests cognitive effort
        if features.energy_mean < 0.06:
            cognitive_load += 0.2
        
        # High pitch variance suggests uncertainty
        if features.pitch_std > 50:
            cognitive_load += 0.1
        
        return min(cognitive_load, 1.0)
    
    def _identify_cultural_markers(self, features: VoiceFeatures) -> Dict[str, float]:
        """Identify cultural markers from voice features"""
        # Simplified cultural marker identification
        # In practice, this would use more sophisticated models
        
        cultural_markers = {
            'american': 0.0,
            'british': 0.0,
            'australian': 0.0,
            'canadian': 0.0,
            'other': 0.0
        }
        
        # Very simplified heuristics
        if 150 < features.pitch_mean < 200 and features.rhythm_regularity > 0.6:
            cultural_markers['american'] = 0.4
        elif features.pitch_mean > 200 and features.pitch_std < 40:
            cultural_markers['british'] = 0.3
        elif features.pitch_mean < 150 and features.energy_mean > 0.08:
            cultural_markers['australian'] = 0.3
        else:
            cultural_markers['other'] = 0.5
        
        return cultural_markers
    
    def _infer_personality_traits(self, features: VoiceFeatures) -> Dict[str, float]:
        """Infer personality traits from voice features"""
        # Simplified personality trait inference
        # Based on voice characteristics
        
        personality_traits = {
            'extraversion': 0.0,
            'agreeableness': 0.0,
            'conscientiousness': 0.0,
            'neuroticism': 0.0,
            'openness': 0.0
        }
        
        # Extraversion: High energy, high pitch, regular rhythm
        if features.energy_mean > 0.08 and features.pitch_mean > 180 and features.rhythm_regularity > 0.6:
            personality_traits['extraversion'] = 0.4
        
        # Agreeableness: Moderate pitch, regular rhythm, low jitter
        if 160 < features.pitch_mean < 200 and features.rhythm_regularity > 0.5 and features.jitter < 0.06:
            personality_traits['agreeableness'] = 0.3
        
        # Conscientiousness: Regular rhythm, low jitter, moderate energy
        if features.rhythm_regularity > 0.7 and features.jitter < 0.05 and 0.06 < features.energy_mean < 0.1:
            personality_traits['conscientiousness'] = 0.3
        
        # Neuroticism: High jitter, irregular rhythm, high pitch variance
        if features.jitter > 0.08 or features.rhythm_regularity < 0.4 or features.pitch_std > 60:
            personality_traits['neuroticism'] = 0.4
        
        # Openness: High pitch variance, moderate energy
        if features.pitch_std > 50 and 0.05 < features.energy_mean < 0.1:
            personality_traits['openness'] = 0.3
        
        return personality_traits

# Example usage
if __name__ == "__main__":
    # Test the voice analyzer
    analyzer = AdvancedVoiceAnalyzer()
    
    # Generate sample audio data (sine wave for testing)
    duration = 2.0  # seconds
    sample_rate = 22050
    frequency = 440  # A4 note
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * frequency * t)
    
    # Extract features
    features = analyzer.extract_comprehensive_features(audio_data)
    print(f"Extracted features: {features}")
    
    # Infer subtext
    subtext = analyzer.infer_subtext(features)
    print(f"Inferred subtext: {subtext}")
