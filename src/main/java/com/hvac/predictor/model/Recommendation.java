package com.hvac.predictor.model;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import java.time.LocalDateTime;
import java.util.List;

@Entity
@Table(name = "recommendations")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Recommendation {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String city;
    private Double currentTemp;
    private Double predictedMax;
    private Integer sliderValue;
    private Double targetAcSetting; // Added this to match Python output

    @Column(columnDefinition = "TEXT")
    private String actionMessage;

    private String impactLevel;

    @ElementCollection
    private List<String> chartLabels;

    @ElementCollection
    private List<Double> chartValues;

    @Builder.Default
    private LocalDateTime createdAt = LocalDateTime.now();
}