package com.hvac.predictor.controller;

import com.hvac.predictor.model.Recommendation;
import com.hvac.predictor.service.RecommendationService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

import java.util.*;

@Controller
public class RecommendationController {

    private static final Logger logger = LoggerFactory.getLogger(RecommendationController.class);
    private final RecommendationService recommendationService;

    public RecommendationController(RecommendationService recommendationService) {
        this.recommendationService = recommendationService;
    }

    @GetMapping("/")
    public String index(Model model) {
        if (!model.containsAttribute("city")) {
            model.addAttribute("city", "Kolkata");
        }
        if (!model.containsAttribute("slider")) {
            model.addAttribute("slider", 50);
        }
        return "index";
    }

    @PostMapping("/get-recommendation")
    public String getRecommendation(@RequestParam String city,
                                    @RequestParam int slider,
                                    RedirectAttributes ra) {
        try {
            if (city == null || city.trim().isEmpty()) {
                ra.addFlashAttribute("error", "City name is required");
                ra.addFlashAttribute("city", "Kolkata");
                ra.addFlashAttribute("slider", slider);
                return "redirect:/";
            }

            if (slider < 0 || slider > 100) {
                ra.addFlashAttribute("error", "Slider value must be between 0 and 100");
                ra.addFlashAttribute("city", city);
                ra.addFlashAttribute("slider", 50);
                return "redirect:/";
            }

            Recommendation recommendation = recommendationService.processAction(city.trim(), slider);

            if (recommendation == null) {
                ra.addFlashAttribute("error", "Failed to generate recommendation");
                ra.addFlashAttribute("city", city);
                ra.addFlashAttribute("slider", slider);
                return "redirect:/";
            }

            ra.addFlashAttribute("success", true);
            ra.addFlashAttribute("recommendation", recommendation);
            ra.addFlashAttribute("city", city.trim());
            ra.addFlashAttribute("slider", slider);

            logger.info("✅ Generated recommendation for {}, ID: {}, Model: {}",
                    city, recommendation.getId(), recommendation.getModelType());

        } catch (Exception e) {
            logger.error("Error generating recommendation", e);
            ra.addFlashAttribute("error", "Error: " + e.getMessage());
            ra.addFlashAttribute("city", city);
            ra.addFlashAttribute("slider", slider);
        }

        return "redirect:/";
    }

    // API endpoint for AJAX calls
    @PostMapping("/api/predict")
    @ResponseBody
    public ResponseEntity<?> predictApi(@RequestBody Map<String, Object> request) {
        try {
            String city = request.get("city") != null ? request.get("city").toString() : "Kolkata";
            int slider = request.get("slider") != null ? ((Number) request.get("slider")).intValue() : 50;

            Recommendation recommendation = recommendationService.processAction(city.trim(), slider);

            Map<String, Object> response = new LinkedHashMap<>();
            response.put("success", true);
            response.put("data", recommendation);
            response.put("timestamp", System.currentTimeMillis());

            return ResponseEntity.ok(response);

        } catch (Exception e) {
            return ResponseEntity.badRequest().body(Map.of("success", false, "error", e.getMessage()));
        }
    }

    // Get latest recommendation for a city
    @GetMapping("/api/recommendations/{city}/latest")
    @ResponseBody
    public ResponseEntity<?> getLatest(@PathVariable String city) {
        Optional<Recommendation> rec = recommendationService.getLatestForCity(city);
        return rec.map(ResponseEntity::ok).orElse(ResponseEntity.notFound().build());
    }

    // Get city history
    @GetMapping("/api/recommendations/{city}/history")
    @ResponseBody
    public ResponseEntity<?> getHistory(@PathVariable String city) {
        List<Recommendation> history = recommendationService.getCityHistory(city);
        return ResponseEntity.ok(history);
    }

    @GetMapping("/health")
    @ResponseBody
    public ResponseEntity<Map<String, Object>> healthCheck() {
        Map<String, Object> health = new LinkedHashMap<>();
        boolean backendAlive = recommendationService.checkBackendHealth();

        health.put("status", backendAlive ? "HEALTHY" : "DEGRADED");
        health.put("springBoot", "Running");
        health.put("pythonBackend", backendAlive ? "Connected" : "Disconnected");
        health.put("backendUrl", recommendationService.getBackendUrl());
        health.put("timestamp", System.currentTimeMillis());

        return ResponseEntity.ok(health);
    }
}