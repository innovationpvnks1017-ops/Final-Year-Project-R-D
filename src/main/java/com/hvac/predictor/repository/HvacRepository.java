package com.hvac.predictor.repository;


import com.hvac.predictor.model.Recommendation;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface HvacRepository extends JpaRepository<Recommendation, Long> {
    // No code needed here! JpaRepository provides .save() and .findAll()
}
