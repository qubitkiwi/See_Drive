// example/lib/models/accidents/accident_rule.dart
import 'dart:math' as math;

import '../hazard/hazard_class.dart';
import '../hazard/hazard_detection.dart';
import 'accident_type.dart';
import 'accident_level.dart';
import 'accident_decision.dart';
import 'package:flutter/foundation.dart';

class ImuSnapshot {
  final int tUs;
  final double ax, ay, az;
  final double gx, gy, gz;
  final double lax, lay, laz;

  const ImuSnapshot({
    required this.tUs,
    required this.ax, required this.ay, required this.az,
    required this.gx, required this.gy, required this.gz,
    required this.lax, required this.lay, required this.laz,
  });

  double get accMag => math.sqrt(ax * ax + ay * ay + az * az);
  double get linAccMag => math.sqrt(lax * lax + lay * lay + laz * laz);
  double get gyroMag => math.sqrt(gx * gx + gy * gy + gz * gz);

  /// (참고용) az 기준 tilt. 이제 Engine에서는 baseline tilt를 씀
  double get tiltDeg {
    final g = accMag;
    if (g < 1e-6) return 0.0;
    final cosTheta = (az / g).clamp(-1.0, 1.0);
    return math.acos(cosTheta) * 180.0 / math.pi;
  }
}

class AccidentRuleEngine {
  static ImuSnapshot? _prev;

  // =========================
  // ✅ 0도 기준 캘리브레이션
  // =========================
  static List<double>? _g0;

  static void resetBaseline() {
    _g0 = null;
  }

  static List<double> _normalize(List<double> v) {
    final mag = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (mag < 1e-6) return [0, 0, 1];
    return [v[0]/mag, v[1]/mag, v[2]/mag];
  }

  static double _tiltFromBaseline(ImuSnapshot imu) {
    if (_g0 == null) return 0.0;
    final g = _normalize([imu.ax, imu.ay, imu.az]);
    final dot = (g[0]*_g0![0] + g[1]*_g0![1] + g[2]*_g0![2])
        .clamp(-1.0, 1.0);
    return math.acos(dot) * 180.0 / math.pi;
  }

  // =========================
  // ✅ evidence streak / cooldown
  // =========================
  static int _minorStreak = 0;
  static int _moderateStreak = 0;
  static int _severeStreak = 0;

  static const int needMinorFrames = 2;
  static const int needModerateFrames = 2;
  static const int needSevereFrames = 1;

  static int _lastDecisionUs = 0;
  static const int cooldownUs = 5000000;

  // =========================
  // ✅ thresholds
  // =========================
  static const double a1 = 2.0;
  static const double a2 = 6.0;
  static const double a3 = 10.0;

  static const double g1 = 2.0;
  static const double g2 = 6.0;
  static const double g3 = 10.0;

  static const double tiltSevereDeg = 70.0;
  static const int hazardWindowUs = 800000;

  static AccidentDecision? decide({
    required List<HazardDetection> hazards,
    required ImuSnapshot imu,
  }) {
    debugPrint("🟡 decide called t=${imu.tUs}");

    _g0 ??= _normalize([imu.ax, imu.ay, imu.az]);
    final tilt = _tiltFromBaseline(imu);

    // --- cooldown ---
    if (_lastDecisionUs != 0 &&
        (imu.tUs - _lastDecisionUs).abs() < cooldownUs) {
      debugPrint("⏸️ cooldown skip");
      _prev = imu;
      return null;
    }

    final prev = _prev;
    _prev = imu;
    if (prev == null) {
      debugPrint("🟠 prev null (first frame)");
      return null;
    }

    // Δ
    final dLinAcc = (imu.linAccMag - prev.linAccMag).abs();
    final dGyro   = (imu.gyroMag   - prev.gyroMag).abs();
    final dLax = (imu.lax - prev.lax).abs();
    final dLay = (imu.lay - prev.lay).abs();
    final dLaz = (imu.laz - prev.laz).abs();

    // recent hazards
    final recentHazards = hazards.where((h) {
      final dt = (imu.tUs - h.tUs).abs();
      return dt <= hazardWindowUs;
    }).toList();

    bool hasHazard(Set<HazardClass> set) =>
        recentHazards.any((h) => set.contains(h.hazard));

    final hasPothole = hasHazard({HazardClass.pothole});
    final hasVehicle = hasHazard({HazardClass.car, HazardClass.truck, HazardClass.bus});
    final hasSoftObj = hasHazard({HazardClass.animal, HazardClass.person});
    final hasHardObj = hasHazard({
      HazardClass.stone, HazardClass.box, HazardClass.garbageBag, HazardClass.constructionSign,
    });

    final hasAnyHazard = recentHazards.isNotEmpty;

    // ✅ severe → moderate → minor 순서로 레벨 결정
    AccidentLevel? levelCandidate;
    if (dLinAcc > a3 || dGyro > g3 || tilt > tiltSevereDeg) {
      levelCandidate = AccidentLevel.severe;
    } else if (dLinAcc > a2 || dGyro > g2) {
      levelCandidate = AccidentLevel.moderate;
    } else if (dLinAcc > a1 || dGyro > g1) {
      levelCandidate = AccidentLevel.minor;
    } else {
      _minorStreak = _moderateStreak = _severeStreak = 0;
      return null;
    }

    // // (실험중이면 유지)
    if (!hasAnyHazard && levelCandidate != AccidentLevel.severe) {
      _minorStreak = _moderateStreak = 0;
      return null;
    }

    // ----- streak accumulate -----
    if (levelCandidate == AccidentLevel.minor) {
      _minorStreak++;
      _moderateStreak = _severeStreak = 0;
      if (_minorStreak < needMinorFrames) return null;
    } else if (levelCandidate == AccidentLevel.moderate) {
      _moderateStreak++;
      _minorStreak = _severeStreak = 0;
      if (_moderateStreak < needModerateFrames) return null;
    } else {
      _severeStreak++;
      _minorStreak = _moderateStreak = 0;
      if (_severeStreak < needSevereFrames) return null;
    }

    final level = levelCandidate;

    // ✅ 후보 수집
    final candidates = <_Cand>[];

    if (level == AccidentLevel.severe &&
        (tilt > tiltSevereDeg || dGyro > g3)) {
      candidates.add(_Cand(
        AccidentType.rollover,
        100,
        "전복/대충격(기울기 ${tilt.toStringAsFixed(1)}°, gyroΔ ${dGyro.toStringAsFixed(2)})",
      ));
    }

    if (hasVehicle &&
        dLinAcc > a2 &&
        (dLax > dLaz || dLay > dLaz)) {
      candidates.add(_Cand(
        AccidentType.collision,
        80,
        "차량 탐지 + 강한 XY 충격",
      ));
    }

    if (hasVehicle && dLay > a1 && dLay > dLax) {
      candidates.add(_Cand(
        AccidentType.sideswipe,
        70,
        "차량 탐지 + 측면(Y) 충격",
      ));
    }

    if (hasPothole && dLaz > a1) {
      candidates.add(_Cand(
        AccidentType.potholeImpact,
        60,
        "포트홀 탐지 + Z 충격",
      ));
    }

    if ((hasHardObj || hasSoftObj) && dLinAcc > a1) {
      candidates.add(_Cand(
        AccidentType.objectImpact,
        50,
        "사물 탐지 + 충격",
      ));
    }

    if (level != AccidentLevel.severe) {
      candidates.add(_Cand(
        AccidentType.contact,
        40,
        hasAnyHazard ? "약한 충격 + 위험요소 동반" : "약한 충격(위험요소 없음)",
      ));
    }

    if (candidates.isEmpty && level == AccidentLevel.severe) {
      candidates.add(_Cand(
        AccidentType.collision,
        10,
        "강충격(severe) 단독 감지",
      ));
    }

    if (candidates.isEmpty) {
      _minorStreak = _moderateStreak = _severeStreak = 0;
      return null;
    }

    candidates.sort((a, b) => b.priority.compareTo(a.priority));
    final chosen = candidates.first;

    final decision = AccidentDecision(
      tUs: imu.tUs,
      type: chosen.type,
      level: level,
      reason: chosen.reason,
      hazards: recentHazards,
      linAccMag: dLinAcc,
      gyroMag: dGyro,
    );

    _lastDecisionUs = imu.tUs;
    _minorStreak = _moderateStreak = _severeStreak = 0;
    return decision;
  }
}

// ✅ decide() 밖, 파일 하단에 private class로 둬야 함.
class _Cand {
  final AccidentType type;
  final int priority;
  final String reason;
  const _Cand(this.type, this.priority, this.reason);
}

// | 사고 타입                     | priority | 의미                               |
// | -------------------------    | -------- | ---------------------------------- |
// | rollover (전복사고)           | **100**  | 가장 심각 → 제일 먼저 선택           |
// | collision (차량 충돌)         | **80**   | 매우 강한 충격                      |
// | sideswipe (측면 충돌)         | **70**   | 특정 방향 충격                      |
// | potholeImpact                | **60**   | 바닥 충격                           |
// | objectImpact                 | **50**   | 사물 충돌                           |
// | contact                      | **40**   | 약한 충격                           |
// | fallback severe collision    | **10**   | severe인데 조건 안 맞을 때 최소 처리 |
