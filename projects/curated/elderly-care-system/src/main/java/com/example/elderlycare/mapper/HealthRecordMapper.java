package com.example.elderlycare.mapper;

import com.example.elderlycare.entity.HealthRecord;
import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface HealthRecordMapper {

    // ==========================================================
    // 🔴 修复重点：把 SQL 里的 measureTime 改成 measure_time
    // ==========================================================
    @Insert("INSERT INTO health_record " +
            "(user_id, high_pressure, low_pressure, blood_sugar, heart_rate, measure_time) " + // 这里必须是数据库列名(下划线)
            "VALUES " +
            "(#{userId}, #{highPressure}, #{lowPressure}, #{bloodSugar}, #{heartRate}, #{measureTime})") // 这里是Java属性名(驼峰)
    void insert(HealthRecord record);

    @Select("SELECT * FROM health_record WHERE user_id = #{userId} ORDER BY measure_time DESC LIMIT 10")
    List<HealthRecord> findRecentByUserId(Long userId);
}