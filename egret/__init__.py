# __init__.py
#
# Copyright (C) 2025 Hirokazu Maesaka (RIKEN SPring-8 Center)
#
# This file is part of Egret: Engine for General Research in
# Energetic-beam Tracking.
#
# Egret is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from .version import __version__

# Lazy import a number of submodules and names to keep `import egret`
# lightweight. Accessing attributes below will import the backing
# submodule on demand. This avoids importing workspace-top-level
# helper modules when the package is only used to access compiled
# bindings via `_pyegret_bridge`.

__all__ = [
	"__version__",
	"Object",
	"Element",
	"Drift",
	"Dipole",
	"Quadrupole",
	"Sextupole",
	"Octupole",
	"Steering",
	"Lattice",
	"Ring",
	"read_ring",
	"Coordinate",
	"CoordinateArray",
	"Envelope",
	"EnvelopeArray",
	"Dispersion",
	"DispersionArray",
]

_LAZY_MAP = {
	'Object': ('.object', 'Object'),
	'Element': ('.element', 'Element'),
	'Drift': ('.drift', 'Drift'),
	'Dipole': ('.dipole', 'Dipole'),
	'Quadrupole': ('.quadrupole', 'Quadrupole'),
	'Sextupole': ('.sextupole', 'Sextupole'),
	'Octupole': ('.octupole', 'Octupole'),
	'Steering': ('.steering', 'Steering'),
	'Lattice': ('.lattice', 'Lattice'),
	'Ring': ('.ring', 'Ring'),
	'read_ring': ('.readring', 'read_ring'),
	'Coordinate': ('.coordinate', 'Coordinate'),
	'CoordinateArray': ('.coordinatearray', 'CoordinateArray'),
	'Envelope': ('.envelope', 'Envelope'),
	'EnvelopeArray': ('.envelopearray', 'EnvelopeArray'),
	'Dispersion': ('.dispersion', 'Dispersion'),
	'DispersionArray': ('.dispersionarray', 'DispersionArray'),
}

def __getattr__(name: str):
	if name in _LAZY_MAP:
		submod, attr = _LAZY_MAP[name]
		module = __import__(f"{__name__}{submod}", fromlist=[attr])
		val = getattr(module, attr)
		globals()[name] = val
		return val
	raise AttributeError(f"module {__name__} has no attribute {name}")

def __dir__():
	return sorted(list(globals().keys()) + list(_LAZY_MAP.keys()))
