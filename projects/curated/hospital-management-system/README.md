# 医院管理系统

## 项目简介

前后端分离的医院信息管理系统（HIS），覆盖管理员、医生、患者三类角色。技术栈为 Spring Boot + MyBatis Plus + MySQL + Vue2 + Element UI + ECharts + JWT 认证。

## 代码架构

### 后端 (`backend/`)

```java
// Spring Boot 2.x + MyBatis Plus + MySQL 8.0

// 入口
BackendApplication.java                    // @SpringBootApplication

// 通用组件
common/R.java                              // 统一响应封装 (code, msg, data)
config/JwtInterceptor.java                 // JWT Token 拦截验证
config/InterceptorConfig.java              // 拦截器注册 + 白名单配置
config/MyBatisPlusConfig.java              // MyBatis Plus 分页插件配置

// 业务 Controller (各模块 REST API)
controller/AdminUserController.java        // 管理员 CRUD
controller/DoctorController.java           // 医生管理
controller/PatientController.java          // 患者管理
controller/ArrangeController.java          // 挂号预约排班
controller/BedController.java              // 床位管理
controller/MedicineController.java         // 药品管理 (库存、处方)
controller/CheckController.java            // 检查项目管理
controller/OrderController.java            // 订单统计
controller/StatisticsController.java       // ECharts 数据可视化接口

// Service → Mapper (MyBatis Plus BaseMapper)
service/ → mapper/
```

### 前端 (`前端/`)

```
Vue2 + Element UI + Vue Router + Axios + ECharts

路由结构:
  /login          → 登录页 (JWT Token 存储)
  /admin/dashboard → 管理员仪表盘 (ECharts 统计面板)
  /admin/doctor    → 医生管理 (CRUD 表格)
  /admin/patient   → 患者管理
  /doctor/schedule → 医生排班
  /doctor/patient  → 就诊记录
  /patient/register→ 患者挂号
  /patient/order   → 订单查询

核心组件:
  - JWT 认证: 登录 → 获取 Token → Axios 拦截器自动附加
  - ECharts: 就诊量折线图、科室占比饼图、收入统计柱状图
  - Element UI: 表格分页、表单验证、对话框确认
```

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | Spring Boot 2.x, MyBatis Plus, MySQL 8.0 |
| 安全 | JWT (JSON Web Token) 无状态认证 |
| 前端 | Vue2, Element UI, Vue Router, Axios |
| 可视化 | ECharts (仪表盘统计图表) |
| 构建 | Maven (pom.xml) |

## 运行方式

```bash
# 后端
cd backend
mvn clean install
java -jar target/hospital-*.jar

# 前端
cd 前端
npm install
npm run serve
```

## 关键文件

| 文件 | 说明 |
|------|------|
| `backend/pom.xml` | Maven 依赖 (Spring Boot, MyBatis Plus, JWT, MySQL) |
| `backend/src/main/java/com/shanzhu/hospital/BackendApplication.java` | 应用入口 |
| `backend/.../config/JwtInterceptor.java` | JWT 拦截器 |
| `backend/.../config/MyBatisPlusConfig.java` | MyBatis Plus 分页配置 |
| `backend/.../controller/` | 全业务 Controller (10+ 模块) |
