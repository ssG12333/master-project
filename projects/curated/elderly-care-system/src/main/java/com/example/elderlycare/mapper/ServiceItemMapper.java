package com.example.elderlycare.mapper;

import com.example.elderlycare.entity.ServiceItem;
import org.apache.ibatis.annotations.*;

import java.util.List;

@Mapper
public interface ServiceItemMapper {

    // 1. 查询所有
    @Select("SELECT * FROM service_item")
    List<ServiceItem> findAll();

    // 2. 根据ID查询 (修复报错用的)
    @Select("SELECT * FROM service_item WHERE id = #{id}")
    ServiceItem findById(Long id);

    // 3. 新增 (这里之前是对的)
    @Insert("INSERT INTO service_item (name, category, price, description, img_url) " +
            "VALUES (#{name}, #{category}, #{price}, #{description}, #{imgUrl})")
    @Options(useGeneratedKeys = true, keyProperty = "id")
    void insert(ServiceItem item);

    // ========================================================
    // 🔴 核心修复：Update 语句加上 img_url = #{imgUrl}
    // ========================================================
    @Update("UPDATE service_item SET " +
            "name=#{name}, " +
            "category=#{category}, " +
            "price=#{price}, " +
            "description=#{description}, " +
            "img_url=#{imgUrl} " +  // <--- 之前就是缺了这一行！
            "WHERE id = #{id}")
    void update(ServiceItem item);

    // 5. 删除
    @Delete("DELETE FROM service_item WHERE id = #{id}")
    void deleteById(Long id);
}