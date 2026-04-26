package com.hvac.predictor.controller;

import com.hvac.predictor.model.Recommendation;
import com.hvac.predictor.service.HvacService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

@Controller
@RequiredArgsConstructor
public class HvacController {

    private final HvacService hvacService;

    @GetMapping("/")
    public String showDashboard() {
        return "index";
    }

    @PostMapping("/get-recommendation")
    public String getRecommendation(@RequestParam String city,
                                    @RequestParam int slider,
                                    Model model) {
        Recommendation result = hvacService.processAction(city, slider);
        model.addAttribute("result", result);
        return "index";
    }
}