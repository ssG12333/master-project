package com.example.elderlycare.entity;

import lombok.Data;

@Data
public class User {
    private Long id;
    private String username;
    private String password;
    private String realName;
    private String role;     // 角色

    // === 新增字段 ===
    private String phone;    // 电话
    private Integer age;     // 年龄
    private String avatar;   // 头像地址
}