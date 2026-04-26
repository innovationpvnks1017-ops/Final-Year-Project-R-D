package com.hvac.predictor.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.hvac.predictor.model.Recommendation;
import com.hvac.predictor.repository.RecommendationRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.client.RestTemplate;

import java.time.LocalDateTime;
import java.util.*;

@Service
public class RecommendationService {

    private static final Logger logger = LoggerFactory.getLogger(RecommendationService.class);
    private final RecommendationRepository recommendationRepository;
    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;

    private static final String PYTHON_BACKEND_URL = "http://localhost:5000";

    public RecommendationService(RecommendationRepository recommendationRepository) {
        this.recommendationRepository = recommendationRepository;
        this.restTemplate = new RestTemplate();
        this.objectMapper = new ObjectMapper();
    }

    @Transactional
    public Recommendation processAction(String city, int slider) {
        try {
            if (city == null || city.trim().isEmpty()) {
                throw new IllegalArgumentException("City name is required");
            }
            if (slider < 0 || slider > 100) {
                throw new IllegalArgumentException("Slider value must be between 0 and 100");
            }

            Recommendation recommendation = callPythonBackend(city, slider);
            recommendation = saveRecommendation(recommendation);

            logger.info("✅ Saved recommendation ID: {} for city: {}", recommendation.getId(), city);
            return recommendation;

        } catch (Exception e) {
            logger.error("Failed to process recommendation for city: {}", city, e);
            return createFallbackRecommendation(city, slider);
        }
    }

    private Recommendation callPythonBackend(String city, int slider) {
        try {
            if (!checkBackendHealth()) {
                throw new RuntimeException("Python backend is not reachable at " + PYTHON_BACKEND_URL);
            }

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            Map<String, Object> requestMap = new HashMap<>();
            requestMap.put("city", city);
            requestMap.put("slider", slider);

            String requestBody = objectMapper.writeValueAsString(requestMap);
            HttpEntity<String> entity = new HttpEntity<>(requestBody, headers);

            logger.info("📡 Sending request to: {}/predict", PYTHON_BACKEND_URL);

            ResponseEntity<Map> response = restTemplate.exchange(
                    PYTHON_BACKEND_URL + "/predict",
                    HttpMethod.POST,
                    entity,
                    Map.class
            );

            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                Map<String, Object> responseBody = response.getBody();

                if (responseBody.containsKey("error")) {
                    throw new RuntimeException("Backend error: " + responseBody.get("error"));
                }

                return mapToRecommendation(responseBody, city, slider);
            } else {
                throw new RuntimeException("Backend returned: " + response.getStatusCode());
            }

        } catch (Exception e) {
            logger.error("Backend communication failed: {}", e.getMessage());
            throw new RuntimeException("Backend communication failed", e);
        }
    }

    @SuppressWarnings("unchecked")
    private Recommendation mapToRecommendation(Map<String, Object> map, String city, int slider) {
        try {
            Recommendation r = new Recommendation();

            // Basic info
            r.setCity(city);
            r.setSliderValue(slider);
            r.setTimestamp(getString(map, "timestamp"));

            // Current conditions
            r.setCurrentTemp(getDouble(map, "current_temp"));
            r.setCurrentHumidity(getDouble(map, "current_humidity"));
            r.setCurrentPressure(getDouble(map, "current_pressure"));
            r.setCurrentWind(getDouble(map, "current_wind"));
            r.setCurrentWeather(getString(map, "current_weather"));
            r.setComfortIndex(getDouble(map, "comfort_index"));
            r.setComfortIndexPeak(getDouble(map, "comfort_index_peak"));

            // Prediction results
            r.setPredictedPeak(getDouble(map, "predicted_peak"));
            r.setPeakHour(getInteger(map, "peak_hour"));
            r.setTempRise(getDouble(map, "temp_rise"));
            r.setTargetAc(getDouble(map, "target_ac"));
            r.setEnergySavingPotential(getDouble(map, "energy_saving_potential"));
            r.setConfidence(getDouble(map, "confidence"));

            // Time series
            r.setLabels(toJson(map.get("labels")));
            r.setTemperatureValues(toJson(map.get("temperature_values")));
            r.setHumidityValues(toJson(map.get("humidity_values")));
            r.setPressureValues(toJson(map.get("pressure_values")));

            // Historical data
            r.setHistoricalLabels(toJson(map.get("historical_labels")));
            r.setHistoricalTemperature(toJson(map.get("historical_temperature")));
            r.setHistoricalHumidity(toJson(map.get("historical_humidity")));
            r.setHistoricalPressure(toJson(map.get("historical_pressure")));

            // Charts (base64 images from Python matplotlib)
            Map<String, Object> charts = (Map<String, Object>) map.get("charts");
            if (charts != null) {
                r.setChartTemperature(getString(charts, "temperature_chart"));
                r.setChartHumidity(getString(charts, "humidity_chart"));
                r.setChartPressure(getString(charts, "pressure_chart"));
                r.setChartShap(getString(charts, "shap_chart"));
                r.setChartDashboard(getString(charts, "dashboard"));
            }

            // SHAP data
            Map<String, Object> shapData = (Map<String, Object>) map.get("shap_data");
            if (shapData != null) {
                r.setShapLabels(toJson(shapData.get("labels")));
                r.setShapPercentages(toJson(shapData.get("percentages")));
                r.setPrimaryDriver(getString(shapData, "primary_driver"));
                r.setTemporalInsights(toJson(shapData.get("temporal_insights")));
            }

            // AI Analysis
            Map<String, Object> aiAnalysis = (Map<String, Object>) map.get("ai_analysis");
            if (aiAnalysis != null) {
                r.setAiSummary(getString(aiAnalysis, "summary"));
                r.setRecommendation(getString(aiAnalysis, "recommendation"));
                r.setEnergyImpact(getString(aiAnalysis, "energy_impact"));
                r.setWeatherAlert(getString(aiAnalysis, "weather_alert"));
            }

            // Technical metrics
            Map<String, Object> techMetrics = (Map<String, Object>) map.get("technical_metrics");
            if (techMetrics != null) {
                r.setModelType(getString(techMetrics, "model_type"));
                r.setDataSource(getString(techMetrics, "data_source"));
                r.setTrainingSamples(getInteger(techMetrics, "training_samples"));
                r.setSequenceLength(getInteger(techMetrics, "sequence_length"));
                r.setFeaturesUsed(getInteger(techMetrics, "features_used"));
                r.setExplainability(getString(techMetrics, "explainability"));
                r.setLastUpdated(getString(techMetrics, "last_updated"));
            }

            r.setImpact(r.determineImpact());
            r.setAiMessage(r.generateAiMessage());

            return r;

        } catch (Exception e) {
            logger.error("Error mapping response", e);
            throw new RuntimeException("Failed to map response", e);
        }
    }

    private String toJson(Object obj) {
        try {
            return obj != null ? objectMapper.writeValueAsString(obj) : null;
        } catch (JsonProcessingException e) {
            return null;
        }
    }

    private Double getDouble(Map<String, Object> map, String key) {
        Object value = map.get(key);
        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        }
        return null;
    }

    private Integer getInteger(Map<String, Object> map, String key) {
        Object value = map.get(key);
        if (value instanceof Number) {
            return ((Number) value).intValue();
        }
        return null;
    }

    private String getString(Map<String, Object> map, String key) {
        Object value = map.get(key);
        return value != null ? value.toString() : null;
    }

    @Transactional
    public Recommendation saveRecommendation(Recommendation recommendation) {
        try {
            return recommendationRepository.save(recommendation);
        } catch (Exception e) {
            logger.error("Database save failed", e);
            throw new RuntimeException("Database save failed", e);
        }
    }

    private Recommendation createFallbackRecommendation(String city, int slider) {
        Recommendation fb = new Recommendation();
        fb.setCity(city);
        fb.setSliderValue(slider);
        fb.setTimestamp(new Date().toString());
        fb.setCurrentTemp(30.0);
        fb.setCurrentHumidity(60.0);
        fb.setCurrentPressure(1013.0);
        fb.setCurrentWind(5.0);
        fb.setCurrentWeather("Unknown");
        fb.setComfortIndex(32.0);
        fb.setComfortIndexPeak(35.0);
        fb.setPredictedPeak(33.0);
        fb.setPeakHour(3);
        fb.setTempRise(3.0);
        fb.setTargetAc(26.0 - (slider / 20.0));
        fb.setEnergySavingPotential(15.0);
        fb.setConfidence(40.0);
        fb.setImpact("MODERATE");
        fb.setEnergyImpact("Moderate");
        fb.setModelType("Fallback - Backend Unavailable");
        fb.setDataSource("None");
        fb.setAiSummary("Python backend unavailable at " + PYTHON_BACKEND_URL + ". Using fallback estimation.");
        fb.setRecommendation("Check Python backend connection");
        fb.setAiMessage("Backend offline - showing estimated values");
        fb.setPrimaryDriver("Temperature");
        fb.setTrainingSamples(0);
        fb.setSequenceLength(24);
        fb.setFeaturesUsed(3);
        fb.setExplainability("None");

        try {
            fb.setLabels("[\"1 AM\",\"2 AM\",\"3 AM\",\"4 AM\",\"5 AM\",\"6 AM\",\"7 AM\",\"8 AM\"]");
            fb.setTemperatureValues("[30,31,32,33,32,31,30,29]");
            fb.setHumidityValues("[60,58,55,52,54,57,59,61]");
            fb.setPressureValues("[1013,1012,1011,1010,1011,1012,1013,1014]");
            fb.setShapLabels("[\"Temperature\",\"Humidity\",\"Pressure\"]");
            fb.setShapPercentages("[45.0,35.0,20.0]");
        } catch (Exception e) {
            logger.warn("Failed to set fallback arrays", e);
        }

        return saveRecommendation(fb);
    }

    public List<Recommendation> getCityHistory(String city) {
        return recommendationRepository.findByCityOrderByCreatedAtDesc(city);
    }

    public Optional<Recommendation> getLatestForCity(String city) {
        return recommendationRepository.findTopByCityOrderByCreatedAtDesc(city);
    }

    public List<Recommendation> getCriticalRecommendations() {
        return recommendationRepository.findByImpact("CRITICAL");
    }

    public List<Recommendation> getRecentRecommendations(int hours) {
        return recommendationRepository.findRecentRecommendations(LocalDateTime.now().minusHours(hours));
    }

    public boolean checkBackendHealth() {
        try {
            ResponseEntity<String> response = restTemplate.getForEntity(
                    PYTHON_BACKEND_URL + "/health", String.class);
            return response.getStatusCode() == HttpStatus.OK;
        } catch (Exception e) {
            return false;
        }
    }

    @Transactional
    public void cleanupOldRecommendations(int daysOld) {
        recommendationRepository.deleteByCreatedAtBefore(LocalDateTime.now().minusDays(daysOld));
    }

    public String getBackendUrl() {
        return PYTHON_BACKEND_URL;
    }
}