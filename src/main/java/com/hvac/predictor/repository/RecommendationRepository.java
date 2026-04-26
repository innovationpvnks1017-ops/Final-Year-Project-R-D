package com.hvac.predictor.repository;

import com.hvac.predictor.model.Recommendation;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
public interface RecommendationRepository extends JpaRepository<Recommendation, Long> {

    List<Recommendation> findByCityOrderByCreatedAtDesc(String city);

    Optional<Recommendation> findTopByCityOrderByCreatedAtDesc(String city);

    List<Recommendation> findByImpact(String impact);

    List<Recommendation> findByTempRiseGreaterThan(Double minTempRise);

    List<Recommendation> findByEnergyImpact(String energyImpact);

    List<Recommendation> findByConfidenceGreaterThanEqual(Double minConfidence);

    @Query("SELECT r FROM Recommendation r WHERE r.createdAt >= :since")
    List<Recommendation> findRecentRecommendations(@Param("since") LocalDateTime since);

    @Query("SELECT r.city, COUNT(r) FROM Recommendation r GROUP BY r.city")
    List<Object[]> countByCity();

    @Query("SELECT r.city, AVG(r.tempRise) FROM Recommendation r GROUP BY r.city")
    List<Object[]> averageTempRiseByCity();

    @Query("SELECT r.city, AVG(r.energySavingPotential) FROM Recommendation r GROUP BY r.city")
    List<Object[]> averageEnergySavingByCity();

    @Query("SELECT r.city, AVG(r.confidence) FROM Recommendation r GROUP BY r.city")
    List<Object[]> averageConfidenceByCity();

    void deleteByCreatedAtBefore(LocalDateTime dateTime);
}