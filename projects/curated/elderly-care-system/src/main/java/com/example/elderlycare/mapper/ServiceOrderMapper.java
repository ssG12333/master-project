package com.example.elderlycare.mapper;

import com.example.elderlycare.entity.ServiceOrder;
import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select; // 记得引入这个
import java.util.List; // 记得引入这个
import org.apache.ibatis.annotations.Update; // 别忘了引入

@Mapper
public interface ServiceOrderMapper {
    // 插入一条新的订单记录
    @Insert("INSERT INTO service_order (user_id, service_id, service_name, price, status, create_time) " +
            "VALUES (#{userId}, #{serviceId}, #{serviceName}, #{price}, #{status}, #{createTime})")
    void insert(ServiceOrder order);
    @Select("SELECT * FROM service_order WHERE user_id = #{userId} ORDER BY create_time DESC")
    List<ServiceOrder> findByUserId(Long userId);
    // 新增：更新订单状态
    @Update("UPDATE service_order SET status = #{status} WHERE id = #{id}")
    void updateStatus(Long id, String status);

    // 新增：查询所有订单（给管理员看）
    @Select("SELECT * FROM service_order ORDER BY create_time DESC")
    List<ServiceOrder> findAll();
}