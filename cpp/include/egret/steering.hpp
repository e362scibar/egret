// steering.hpp
#pragma once

#include <Eigen/Dense>
#include "egret/drift.hpp"

namespace egret {

class Steering : public Drift {
public:
    // Tilted kick handling belongs to higher-level element, but basic transport is drift-like
    // We'll reuse Drift implementations; additional helpers could be added here.
};

} // namespace egret
