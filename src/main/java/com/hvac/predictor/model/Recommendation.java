package com.hvac.predictor.model;

import jakarta.persistence.*;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import lombok.Builder;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.time.LocalDateTime;

@Entity
@Table(name = "recommendations")
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Recommendation {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    // Input Parameters
    private String city;

    @Column(columnDefinition = "VARCHAR(100)")
    private String timestamp;

    @Column(name = "slider_value")
    private int sliderValue;

    // Current Conditions
    @JsonProperty("current_temp")
    @Column(name = "current_temp")
    private Double currentTemp;

    @JsonProperty("current_humidity")
    @Column(name = "current_humidity")
    private Double currentHumidity;

    @JsonProperty("current_pressure")
    @Column(name = "current_pressure")
    private Double currentPressure;

    @JsonProperty("current_wind")
    @Column(name = "current_wind")
    private Double currentWind;

    @JsonProperty("current_weather")
    @Column(name = "current_weather", length = 100)
    private String currentWeather;

    @JsonProperty("comfort_index")
    @Column(name = "comfort_index")
    private Double comfortIndex;

    @JsonProperty("comfort_index_peak")
    @Column(name = "comfort_index_peak")
    private Double comfortIndexPeak;

    // Prediction Results
    @JsonProperty("predicted_peak")
    @Column(name = "predicted_peak")
    private Double predictedPeak;

    @JsonProperty("peak_hour")
    @Column(name = "peak_hour")
    private Integer peakHour;

    @JsonProperty("temp_rise")
    @Column(name = "temp_rise")
    private Double tempRise;

    @JsonProperty("target_ac")
    @Column(name = "target_ac")
    private Double targetAc;

    @JsonProperty("energy_saving_potential")
    @Column(name = "energy_saving_potential")
    private Double energySavingPotential;

    private Double confidence;

    // Time series data - using TEXT to avoid row size limit
    @Column(columnDefinition = "TEXT")
    private String labels;

    @Column(name = "temperature_values", columnDefinition = "TEXT")
    @JsonProperty("temperature_values")
    private String temperatureValues;

    @Column(name = "humidity_values", columnDefinition = "TEXT")
    @JsonProperty("humidity_values")
    private String humidityValues;

    @Column(name = "pressure_values", columnDefinition = "TEXT")
    @JsonProperty("pressure_values")
    private String pressureValues;

    // Historical data - using TEXT
    @Column(name = "historical_labels", columnDefinition = "TEXT")
    private String historicalLabels;

    @Column(name = "historical_temperature", columnDefinition = "TEXT")
    private String historicalTemperature;

    @Column(name = "historical_humidity", columnDefinition = "TEXT")
    private String historicalHumidity;

    @Column(name = "historical_pressure", columnDefinition = "TEXT")
    private String historicalPressure;

    // Charts as Base64 images
    @Column(name = "chart_temperature", columnDefinition = "MEDIUMTEXT")
    private String chartTemperature;

    @Column(name = "chart_humidity", columnDefinition = "MEDIUMTEXT")
    private String chartHumidity;

    @Column(name = "chart_pressure", columnDefinition = "MEDIUMTEXT")
    private String chartPressure;

    @Column(name = "chart_shap", columnDefinition = "MEDIUMTEXT")
    private String chartShap;

    @Column(name = "chart_dashboard", columnDefinition = "MEDIUMTEXT")
    private String chartDashboard;

    // SHAP Data
    @Column(name = "shap_labels", columnDefinition = "TEXT")
    private String shapLabels;

    @Column(name = "shap_percentages", columnDefinition = "TEXT")
    private String shapPercentages;

    @JsonProperty("primary_driver")
    @Column(name = "primary_driver", length = 100)
    private String primaryDriver;

    @Column(name = "temporal_insights", columnDefinition = "TEXT")
    private String temporalInsights;

    // AI Analysis
    @Column(name = "ai_summary", columnDefinition = "TEXT")
    private String aiSummary;

    @Column(length = 500)
    private String recommendation;

    @Column(name = "energy_impact", length = 50)
    private String energyImpact;

    @Column(name = "weather_alert", length = 500)
    private String weatherAlert;

    // Technical Metrics
    @Column(name = "model_type", length = 200)
    private String modelType;

    @Column(name = "data_source", length = 100)
    private String dataSource;

    @Column(name = "training_samples")
    private Integer trainingSamples;

    @Column(name = "sequence_length")
    private Integer sequenceLength;

    @Column(name = "features_used")
    private Integer featuresUsed;

    @Column(length = 100)
    private String explainability;

    @Column(name = "last_updated", length = 50)
    private String lastUpdated;

    // Metadata
    @Column(name = "created_at")
    private LocalDateTime createdAt;

    @Column(length = 50)
    private String impact;

    @Column(name = "ai_message", columnDefinition = "TEXT")
    private String aiMessage;

    @PrePersist
    protected void onCreate() {
        if (createdAt == null) {
            createdAt = LocalDateTime.now();
        }
    }

    public String determineImpact() {
        if (tempRise != null && tempRise > 3.0) {
            return "CRITICAL";
        }
        return "MODERATE";
    }

    public String generateAiMessage() {
        if (primaryDriver != null && confidence != null) {
            return String.format("BiLSTM analysis: Primary driver is %s with %.1f%% confidence",
                    primaryDriver, confidence);
        }
        return "AI analysis in progress";
    }
}