package com.example.elderlycare.mapper;

import com.example.elderlycare.entity.User;
import org.apache.ibatis.annotations.*;

import java.util.List;

@Mapper
public interface UserMapper {

    // 1. 登录查账号 (保留)
    @Select("SELECT * FROM sys_user WHERE username = #{username}")
    User findByUsername(String username);

    // 2. 注册/新增 (保留)
    @Insert("INSERT INTO sys_user (username, password, real_name, role, phone, age, avatar) " +
            "VALUES (#{username}, #{password}, #{realName}, #{role}, #{phone}, #{age}, #{avatar})")
    @Options(useGeneratedKeys = true, keyProperty = "id")
    void insert(User user);

    // 3. 更新资料 (保留)
    @Update("UPDATE sys_user SET real_name=#{realName}, phone=#{phone}, age=#{age}, avatar=#{avatar}, role=#{role} WHERE id = #{id}")
    void update(User user);

    // ==========================================
    // 🆕 新增：管理员用的功能
    // ==========================================

    // 4. 查所有用户
    @Select("SELECT * FROM sys_user")
    List<User> findAll();

    // 5. 删除用户
    @Delete("DELETE FROM sys_user WHERE id = #{id}")
    void deleteById(Long id);
}