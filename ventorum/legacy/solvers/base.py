# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""
Abstract base class for all LLT solvers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ventorum.legacy.core.datatypes import Aircraft, FlightCondition, SolverResult, SolverSettings


class BaseSolver(ABC):
    """Interface that every solver must implement."""

    @abstractmethod
    def solve(
        self,
        aircraft: Aircraft,
        condition: FlightCondition,
        settings: SolverSettings,
    ) -> SolverResult:
        """Run the LLT analysis and return results.

        Parameters
        ----------
        aircraft : Aircraft
            Geometry definition (one or more lifting surfaces).
        condition : FlightCondition
            Free-stream conditions.
        settings : SolverSettings
            Discretisation and convergence parameters.

        Returns
        -------
        SolverResult
        """
        ...
