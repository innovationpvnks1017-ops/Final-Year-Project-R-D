package com.hvac.predictor.service;

import com.hvac.predictor.model.Recommendation;
import com.hvac.predictor.repository.HvacRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import java.util.*;

@Service
@RequiredArgsConstructor
public class HvacService {
    private final HvacRepository repository;
    private final RestTemplate restTemplate = new RestTemplate();

    public Recommendation processAction(String city, int slider) {
        String url = "http://localhost:5000/predict";
        Map<String, Object> body = Map.of("city", city, "slider", slider);

        try {
            Map<String, Object> res = restTemplate.postForObject(url, body, Map.class);

            if (res == null) throw new RuntimeException("ML Server returned null");

            // Extract the target AC setting safely
            Double targetAc = Double.valueOf(res.get("target_ac_setting").toString());

            Recommendation rec = Recommendation.builder()
                    .city((String) res.get("city"))
                    .currentTemp(Double.valueOf(res.get("current_temp").toString()))
                    .predictedMax(Double.valueOf(res.get("predicted_peak").toString()))
                    .sliderValue(slider)
                    .targetAcSetting(targetAc)
                    .actionMessage("Nexus AI Recommendation: Maintain AC at " + targetAc + "°C for optimized energy-to-comfort ratio.")
                    .impactLevel((String) res.get("impact_level"))
                    .chartLabels((List<String>) res.get("labels"))
                    .chartValues((List<Double>) res.get("values"))
                    .build();

            return repository.save(rec);
        } catch (Exception e) {
            // Return a fallback object instead of throwing an error to avoid Whitelabel 500
            return Recommendation.builder()
                    .city(city)
                    .actionMessage("Error: ML Service Synchronization Failed. Check if Python server is running.")
                    .impactLevel("UNKNOWN")
                    .currentTemp(0.0)
                    .predictedMax(0.0)
                    .targetAcSetting(0.0)
                    .build();
        }
    }
}