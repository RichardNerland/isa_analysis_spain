import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple, Any

# Only import these when needed in main()
# import matplotlib.pyplot as plt
# import argparse

class Year:
    """
    Simplified class for tracking economic parameters for each simulation year.
    """
    def __init__(self, initial_inflation_rate: float, initial_unemployment_rate: float, 
                 initial_isa_cap: float, initial_isa_threshold: float, num_years: int):
        self.year_count = 1
        self.inflation_rate = initial_inflation_rate
        self.stable_inflation_rate = initial_inflation_rate
        self.unemployment_rate = initial_unemployment_rate
        self.stable_unemployment_rate = initial_unemployment_rate
        self.isa_cap = initial_isa_cap
        self.isa_threshold = initial_isa_threshold
        self.deflator = 1.0
        # Store random seed for reproducibility if needed
        self.random_seed = None

    def next_year(self, random_seed: Optional[int] = None) -> None:
        """
        Advance to the next year and update economic conditions.
        
        Args:
            random_seed: Optional seed for random number generation for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            self.random_seed = random_seed
            
        self.year_count += 1
        
        # More realistic inflation model with bounds
        inflation_shock = np.random.normal(0, 0.01)
        self.inflation_rate = (
            self.stable_inflation_rate * 0.45 + 
            self.inflation_rate * 0.5 + 
            inflation_shock
        )
        # Ensure inflation stays within reasonable bounds
        self.inflation_rate = max(-0.02, min(0.15, self.inflation_rate))
        
        # More realistic unemployment model with bounds
        unemployment_shock = np.random.lognormal(0, 1) / 100
        self.unemployment_rate = (
            self.stable_unemployment_rate * 0.33 + 
            self.unemployment_rate * 0.25 + 
            unemployment_shock
        )
        # Ensure unemployment stays within reasonable bounds
        self.unemployment_rate = max(0.02, min(0.30, self.unemployment_rate))
        
        # Update ISA parameters with inflation
        self.isa_cap *= (1 + self.inflation_rate)
        self.isa_threshold *= (1 + self.inflation_rate)
        self.deflator *= (1 + self.inflation_rate)


class Student:
    """
    Simplified class for tracking student payments without debt seniority.
    """
    def __init__(self, degree, num_years: int):
        self.degree = degree # Current degree
        self.num_years = num_years
        self.earnings_power = 0.0
        self.earnings = [0.0] * num_years
        self.payments = [0.0] * num_years
        self.real_payments = [0.0] * num_years
        self.is_graduated = False
        self.is_employed = False # Employed in the host country
        # self.is_home = False # REMOVED - handled by STAY_HOME degree
        self.is_active = False  # New flag to track active status
        self.years_paid = 0
        self.hit_cap = False
        self.cap_value_when_hit = 0.0  # Store the actual cap value when the student hits it
        self.years_experience = 0 # Years of experience in the *current* role/degree
        self.graduation_year = degree.years_to_complete # Initial graduation year
        self.limit_years = num_years
        self.last_payment_year = -1  # Track the last year a payment was made
        # Promotion tracking
        self.promotion_degree: Optional[Degree] = None
        self.promotion_year: Optional[int] = None


class Degree:
    """
    Class representing different degree options with associated parameters.
    
    Attributes:
        name: Identifier for the degree type (e.g., 'NHA_ECU', 'HOSP_ENTRY', 'STAY_HOME')
        mean_earnings: Average annual earnings for graduates with this degree
        stdev: Standard deviation of earnings for this degree type
        experience_growth: Annual percentage growth in earnings due to experience
        years_to_complete: Number of years required to complete the degree phase (in host country)
        # leave_labor_force_probability: (REMOVED - replaced by STAY_HOME degree)
        promotes_to_degree_name: Optional name of the degree this one promotes to
        years_to_promotion: Optional number of years after graduation until promotion
    """
    def __init__(self, name: str, mean_earnings: float, stdev: float, 
                 experience_growth: float, years_to_complete: int, 
                 # leave_labor_force_probability: float, # REMOVED
                 promotes_to_degree_name: Optional[str] = None,
                 years_to_promotion: Optional[int] = None):
        self.name = name
        self.mean_earnings = mean_earnings
        self.stdev = stdev
        self.experience_growth = experience_growth
        self.years_to_complete = years_to_complete
        # self.leave_labor_force_probability = leave_labor_force_probability # REMOVED
        self.promotes_to_degree_name = promotes_to_degree_name
        self.years_to_promotion = years_to_promotion
    
    def __repr__(self) -> str:
        """String representation of the Degree object for debugging."""
        promo_info = ""
        if self.promotes_to_degree_name and self.years_to_promotion is not None:
            promo_info = f", promotes_to='{self.promotes_to_degree_name}' after {self.years_to_promotion} years"
        # Format leave_labor_force_probability if it exists (it doesn't now)
        # leave_prob_info = f", leave_prob={self.leave_labor_force_probability:.1%}" if hasattr(self, 'leave_labor_force_probability') else ""
        return (f"Degree(name='{self.name}', mean_earnings={self.mean_earnings}, "
                f"stdev={self.stdev}, growth={self.experience_growth:.2%}, "
                f"years={self.years_to_complete}{promo_info})")


def _calculate_graduation_delay(base_years_to_complete: int, degree_name: str = '') -> int:
    """
    Calculate a realistic graduation delay based on degree-specific distributions.
    
    For BA and ASST degrees (Using as default now):
    - 50% graduate on time (no delay)
    - 25% graduate 1 year late (50% of remaining)
    - 12.5% graduate 2 years late (50% of remaining)
    - 6.25% graduate 3 years late (50% of remaining)
    - The rest (6.25%) graduate 4 years late
    
    For MA, NURSE, and TRADE degrees (Using NHA_ECU, NA_ECU, HOSP_ADV as proxy):
    - 75% graduate on time (no delay)
    - 20% graduate 1 year late
    - 2.5% graduate 2 years late
    - 2.5% graduate 3 years late
    
    Args:
        base_years_to_complete: The nominal years to complete the degree phase
        degree_name: The type of degree (NHA_ECU, HOSP_ENTRY, etc.)
        
    Returns:
        Total years to complete including delay
    """
    rand = np.random.random()
    
    # Apply special distribution for degrees assumed similar to MA/Nurse/Trade
    if degree_name in ['NHA_ECU', 'NA_ECU', 'HOSP_ADV']: # Assume these have higher on-time completion
        if rand < 0.75:
            return base_years_to_complete  # Graduate on time
        elif rand < 0.95:
            return base_years_to_complete + 1  # 1 year late
        elif rand < 0.975:
            return base_years_to_complete + 2  # 2 years late
        else:
            return base_years_to_complete + 3  # 3 years late
    else:
        # Default distribution for other degrees (HOSP_ENTRY, etc.)
        if rand < 0.5:
            return base_years_to_complete  # Graduate on time
        elif rand < 0.75:
            return base_years_to_complete + 1  # 1 year late
        elif rand < 0.875:
            return base_years_to_complete + 2  # 2 years late
        elif rand < 0.9375:
            return base_years_to_complete + 3  # 3 years late
        else:
            return base_years_to_complete + 4  # 4 years late


def simulate_simple(
    students: List[Student], 
    year: Year, 
    num_years: int, 
    isa_percentage: float, 
    limit_years: int, 
    performance_fee_pct: float = 0.025,  # 2.5% performance fee on repayments
    gamma: bool = False, # Keep flag, though current logic sets to False
    price_per_student: float = 30000, # Default (will be overridden)
    new_malengo_fee: bool = True,
    annual_fee_per_student: float = 300,  # $300 base annual fee per active student
    apply_graduation_delay: bool = False # Keep flag
) -> Dict[str, Any]:
    """
    Run a single simulation for the given students over the specified number of years
    with a simple repayment structure, handling promotions and STAY_HOME degrees.
    
    The Malengo fee structure consists of:
    1. Annual fee per active student ($300 base, adjusted for inflation)
    2. Performance fee (2.5%) on all student repayments
    """
    # Initialize arrays to track payments
    total_payments = np.zeros(num_years)
    total_real_payments = np.zeros(num_years)
    malengo_payments = np.zeros(num_years)
    malengo_real_payments = np.zeros(num_years)
    investor_payments = np.zeros(num_years)
    investor_real_payments = np.zeros(num_years)
    
    # Track active students for each year
    active_students_count = np.zeros(num_years, dtype=int)
    
    # Track student status for fee calculations
    student_graduated = np.zeros(len(students), dtype=bool) # Tracks graduation from *initial* phase
    student_hit_cap = np.zeros(len(students), dtype=bool)
    # Tracks if student is in a non-paying category (STAY_HOME or potentially NA_ECU if treated differently)
    # Let's use degree name directly instead of this flag where needed.
    # student_is_na = np.zeros(len(students), dtype=bool) # Potentially remove
    
    # Store the limit_years in each student object
    for student in students:
        student.limit_years = limit_years
    
    # Simulation loop
    for i in range(num_years):
        # Process each student
        for student_idx, student in enumerate(students):
            # Reset active status at start of year
            student.is_active = False
            
            # --- Promotion Check ---
            if student.promotion_year is not None and i == student.promotion_year and student.promotion_degree is not None:
                # Promote student
                # print(f"DEBUG: Promoting student {student_idx} from {student.degree.name} to {student.promotion_degree.name} in year {i}") # Optional debug
                previous_degree_name = student.degree.name
                student.degree = student.promotion_degree
                # Reset promotion tracking
                student.promotion_year = None
                student.promotion_degree = None
                # Reset years_experience for the new role
                student.years_experience = 0
                # Recalculate earnings power for the new degree
                new_mean = student.degree.mean_earnings
                new_stdev = student.degree.stdev
                if gamma: 
                    # Placeholder if gamma needed - requires parameter transformation
                    # student.earnings_power = max(0, np.random.gamma(shape, scale)) 
                    student.earnings_power = max(0, np.random.normal(new_mean, new_stdev)) # Fallback to normal
                else: 
                    student.earnings_power = max(0, np.random.normal(new_mean, new_stdev))
                # Mark as graduated from the *new* phase implicitly? No, is_graduated tracks initial.

            # Skip host-country simulation steps if student hasn't completed initial phase yet (and isn't STAY_HOME)
            if i < student.graduation_year and student.degree.name != 'STAY_HOME':
                continue
                
            # Handle initial graduation year (only relevant for non-STAY_HOME)
            # Ensures earnings power is set upon completing initial phase
            if i == student.graduation_year and not student.is_graduated and student.degree.name != 'STAY_HOME':
                _process_graduation(student, student_idx, student_graduated, gamma) 

            # --- Handle STAY_HOME degree ---
            if student.degree.name == 'STAY_HOME':
                student.is_employed = False # Not employed in host country
                student.is_graduated = False # Not graduated in host country context
                # student_is_na[student_idx] = True # No longer needed?
                # Earn fixed local income (affected by host country inflation for threshold/cap comparison)
                if student.earnings_power == 0.0: # Set initial power if not already set
                     student.earnings_power = max(0, np.random.normal(student.degree.mean_earnings, student.degree.stdev))
                student.earnings[i] = student.earnings_power * year.deflator 
                student.years_experience = 0 # No experience growth
                # No payments expected
                student.payments[i] = 0
                student.real_payments[i] = 0
                student.is_active = False # Never active for fee purposes
                continue # Skip rest of the loop for STAY_HOME

            # --- Regular processing for migrated students ---
            # Determine employment status in host country
            _update_employment_status(student, year)
            
            # Process employed students
            if student.is_employed:
                # Ensure earnings_power was set at graduation or promotion
                if student.earnings_power == 0.0 and student.is_graduated: # Safety check
                     print(f"Warning: Student {student_idx} graduated but has 0 earnings power. Setting based on current degree: {student.degree.name}")
                     student.earnings_power = max(0, np.random.normal(student.degree.mean_earnings, student.degree.stdev))

                # Update earnings based on experience in current role
                student.earnings[i] = student.earnings_power * year.deflator * (1 + student.degree.experience_growth) ** student.years_experience
                student.years_experience += 1
                
                # Process payments if earnings exceed threshold
                if student.earnings[i] > year.isa_threshold:
                    # Check if student already hit payment cap (from previous years)
                    if student.hit_cap:
                        continue # Skip payment calculation if cap already hit

                    # Check if student has reached payment year limit
                    if student.years_paid >= student.limit_years: # Use student.limit_years
                        # Mark as hit cap only if not already marked? Redundant check?
                        # student_hit_cap[student_idx] = True # This array seems unused elsewhere, use student.hit_cap
                        continue # Skip payment calculation if years limit reached this year

                    # Increment years paid *if* payment is positive and limits not hit
                    student.years_paid += 1

                    # Calculate potential payment
                    potential_payment = isa_percentage * student.earnings[i]
                    current_total_payments = np.sum(student.payments[:i]) # Sum payments *before* this year

                    # Check if payment would exceed cap
                    if (current_total_payments + potential_payment) >= year.isa_cap:
                        # Payment is the remaining amount to hit the cap
                        student.payments[i] = max(0, year.isa_cap - current_total_payments) # Ensure non-negative payment
                        student.real_payments[i] = student.payments[i] / year.deflator
                        student.hit_cap = True
                        # student_hit_cap[student_idx] = True # Unused array
                        student.cap_value_when_hit = year.isa_cap # Store the cap value at time of hitting
                    else:
                        # Payment is the full potential payment
                        student.payments[i] = potential_payment
                        student.real_payments[i] = potential_payment / year.deflator
                        student.last_payment_year = i # Update last payment year
                    
                    # Add to total payments for the year
                    total_payments[i] += student.payments[i]
                    total_real_payments[i] += student.real_payments[i]
                
                else: # Employed but below threshold
                     student.payments[i] = 0
                     student.real_payments[i] = 0

            else: # Unemployed in host country
                student.years_experience = max(0, student.years_experience - 3) # Existing logic
                student.earnings[i] = 0 # Ensure zero earnings
                student.payments[i] = 0
                student.real_payments[i] = 0
            
            # Update active status for fee calculation (only for graduated, non-capped, non-STAY_HOME)
            # Use initial graduation year stored in student.graduation_year
            # is_na_equivalent = student.degree.name in ['NA_ECU'] # Example if NA treated differently for fees. Assume not for now.
            if student.is_graduated and not student.hit_cap and student.years_paid < student.limit_years: 
                 # Student is active if they made a payment in the last 3 years OR graduated recently
                recent_payment = (i - student.last_payment_year <= 3) if student.last_payment_year >= 0 else False
                # Check against initial graduation year
                recent_graduate = (i - student.graduation_year <= 3) 
                student.is_active = recent_payment or recent_graduate
        
        # Count active students for this year (after processing all students)
        active_students_count[i] = sum(1 for student in students if student.is_active)
        
        # Calculate Malengo's fees using new structure:
        # 1. Annual fee per active student (adjusted for inflation)
        # 2. Performance fee on all repayments made *this year*
        annual_fee_inflated = annual_fee_per_student * year.deflator  # Adjust annual fee for inflation
        active_student_fees = active_students_count[i] * annual_fee_inflated
        performance_fees = total_payments[i] * performance_fee_pct # Based on sum of payments[i] across students
        
        # Total Malengo fees (nominal)
        malengo_payments[i] = active_student_fees + performance_fees
        
        # Real (inflation-adjusted) Malengo fees
        malengo_real_payments[i] = malengo_payments[i] / year.deflator # Adjust total nominal fee
        
        # Calculate investor payments (total payments minus Malengo fees)
        investor_payments[i] = total_payments[i] - malengo_payments[i]
        investor_real_payments[i] = total_real_payments[i] - malengo_real_payments[i]

        # Advance to next year
        year.next_year()

    # Prepare and return results
    data = {
        'Student': students,
        'Degree': [student.degree for student in students], # Tracks final degree if promoted
        'Earnings': [student.earnings for student in students],
        'Payments': [student.payments for student in students],
        'Real_Payments': [student.real_payments for student in students],
        'Total_Payments': total_payments,
        'Total_Real_Payments': total_real_payments,
        'Malengo_Payments': malengo_payments,
        'Malengo_Real_Payments': malengo_real_payments,
        'Investor_Payments': investor_payments,
        'Investor_Real_Payments': investor_real_payments,
        'Active_Students_Count': active_students_count
    }

    return data


def _process_graduation(student: Student, student_idx: int, 
                       student_graduated: np.ndarray, # Removed student_is_na
                       gamma: bool) -> None:
    """Helper function to process a student's graduation from their *initial* degree phase."""
    if student.degree.name == 'STAY_HOME': # Should not be called
        return 

    student.is_graduated = True
    student_graduated[student_idx] = True
    # student_is_na[student_idx] = (student.degree.name == 'NA_ECU') # Removed

    # Removed is_home logic
    
    # Set initial earnings power based on the *initial* degree
    initial_mean = student.degree.mean_earnings
    initial_stdev = student.degree.stdev
    if gamma:
        # Placeholder for gamma calculation if needed
        student.earnings_power = max(0, np.random.normal(initial_mean, initial_stdev))
    else:
        student.earnings_power = max(0, np.random.normal(initial_mean, initial_stdev))
        
    # Removed adjustment for students who return home


def _update_employment_status(student: Student, year: Year) -> None:
    """Helper function to update a student's employment status in the host country."""
    # STAY_HOME students are never employed in the host country
    if student.degree.name == 'STAY_HOME':
        student.is_employed = False
    elif year.unemployment_rate < 1: # Check unemployment rate is valid
         # Employed based on inverse of unemployment rate
        student.is_employed = np.random.binomial(1, 1 - year.unemployment_rate) == 1
    else: # Unemployment rate is 100% or invalid
        student.is_employed = False


# Remove unused helper _calculate_malengo_fees
# def _calculate_malengo_fees(...): ...


def _setup_degree_distribution(
    program_type: str, 
    base_degrees: Dict[str, Dict[str, Any]], # Pass definitions dict
    # leave_labor_force_probability: float, # REMOVED
    # Ecuador program parameters
    ecu_year1_completion_prob: float = 0.90,
    ecu_placement_prob: float = 0.80,
    ecu_na_completion_prob: float = 0.85, # Chance to promote NHA->NA given placement
    # Guatemala program parameters
    guat_placement_prob: float = 0.85,
    guat_advancement_prob: float = 0.40, # Chance to promote ENTRY->ADV given placement
    # Custom degree percentages (REMOVED)
) -> Tuple[List[Degree], List[float]]:
    """
    Helper function to set up *initial* degree distribution based on program type and pathways.
    Determines the starting degree (e.g., NHA_ECU, HOSP_ENTRY, STAY_HOME).
    Promotion logic is handled later based on Degree attributes and further randomization.
    
    Returns:
         Tuple[List of possible initial Degree objects, List of corresponding probabilities]
    """
    initial_degrees = []
    initial_probs = []
    
    # Create Degree objects from definitions for lookup
    degree_objects = {name: Degree(**params) for name, params in base_degrees.items()}
    stay_home_degree = degree_objects['STAY_HOME']
    
    if program_type == 'Ecuador':
        # Paths lead to an initial state: STAY_HOME or NHA_ECU (start of host country phase)
        prob_stay_home = 1.0 - ecu_year1_completion_prob
        # All who complete year 1 initially start the host country phase aiming for NHA_ECU.
        # Promotion to NA_ECU happens later *if* they pass placement and NA completion checks.
        prob_start_nha = ecu_year1_completion_prob
        
        initial_degrees = [degree_objects['NHA_ECU'], stay_home_degree]
        initial_probs = [prob_start_nha, prob_stay_home]
        
    elif program_type == 'Guatemala':
        # Paths lead to an initial state: STAY_HOME or HOSP_ENTRY (start of host country phase)
        prob_stay_home = 1.0 - guat_placement_prob
        # All who are placed start as HOSP_ENTRY. Promotion happens later if they pass advancement check.
        prob_start_entry = guat_placement_prob

        initial_degrees = [degree_objects['HOSP_ENTRY'], stay_home_degree]
        initial_probs = [prob_start_entry, prob_stay_home]
        
    else:
        raise ValueError("Program type must be 'Ecuador' or 'Guatemala'")
    
    # Normalize probabilities (should sum to 1 already, but good practice)
    prob_sum = sum(initial_probs)
    if prob_sum > 0:
        initial_probs = [p / prob_sum for p in initial_probs]
    elif len(initial_probs) > 0: # Avoid division by zero if list is not empty but sums to 0
        # This case should not happen with valid inputs
        raise ValueError("Sum of probabilities is zero for non-empty list.")
    # If initial_probs is empty (e.g., invalid program type), error was already raised.
        
    return initial_degrees, initial_probs


def run_simple_simulation(
    program_type: str, # No default
    num_students: int,
    num_sims: int,
    # scenario: str = 'baseline', # REMOVED
    initial_unemployment_rate: float = 0.08,
    initial_inflation_rate: float = 0.02,
    performance_fee_pct: float = 0.025,
    # leave_labor_force_probability: float = 0.05, # REMOVED
    # Ecuador program parameters
    ecu_year1_completion_prob: float = 0.90,
    ecu_placement_prob: float = 0.80,
    ecu_na_completion_prob: float = 0.85, # Chance NHA->NA given placement
    # Guatemala program parameters
    guat_placement_prob: float = 0.85,
    guat_advancement_prob: float = 0.40, # Chance ENTRY->ADV given placement
    # Spain program parameters (REMOVED)
    # ISA parameters
    isa_percentage: Optional[float] = None,
    isa_threshold: float = 13_000,
    isa_cap: Optional[float] = None,
    price_per_student: Optional[float] = None,
    new_malengo_fee: bool = True,
    annual_fee_per_student: float = 300,
    # Additional parameters
    random_seed: Optional[int] = None,
    num_years: int = 25,
    limit_years: int = 10,
    apply_graduation_delay: bool = False
) -> Dict[str, Any]:
    """
    Run multiple simulations for Ecuador or Guatemala programs with promotions and STAY_HOME path.
    """
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Validate program type
    if program_type not in ['Ecuador', 'Guatemala']:
        raise ValueError("Program type must be 'Ecuador' or 'Guatemala'")

    # Determine program-specific defaults IF parameters are not provided
    if price_per_student is None:
        if program_type == 'Ecuador':
            price_per_student = 9_000
        elif program_type == 'Guatemala':
            price_per_student = 7_500 

    if isa_percentage is None:
        isa_percentage = 0.10 # Default 10% for both

    if isa_cap is None:
        if program_type == 'Ecuador':
            isa_cap = 27_000 # 3 * 9k
        elif program_type == 'Guatemala':
            isa_cap = 22_500 # 3 * 7.5k

    # Define all possible degree types from definitions
    base_degrees_dict = _create_degree_definitions()
    # Convert dict values to Degree objects for easier handling in _create_students
    base_degrees_all = {name: Degree(**params) for name, params in base_degrees_dict.items()}
    
    # Set up initial degree distribution based on program path probabilities
    initial_degrees_list, initial_probs = _setup_degree_distribution(
        program_type=program_type,
        base_degrees=base_degrees_dict, # Pass the definition dict
        ecu_year1_completion_prob=ecu_year1_completion_prob,
        ecu_placement_prob=ecu_placement_prob,
        ecu_na_completion_prob=ecu_na_completion_prob,
        guat_placement_prob=guat_placement_prob,
        guat_advancement_prob=guat_advancement_prob
    )
    
    # Prepare containers for results
    total_payment = {}
    investor_payment = {}
    malengo_payment = {}
    nominal_total_payment = {}
    nominal_investor_payment = {}
    nominal_malengo_payment = {}
    active_students = {}
    df_list = []
    employment_stats = []
    ever_employed_stats = []
    repayment_stats = []
    cap_stats = []
    
    # Calculate total investment
    total_investment = num_students * price_per_student
    
    # Run multiple simulations
    for trial in range(num_sims):
        # Initialize year class with a unique seed for each trial if random_seed is provided
        trial_seed = random_seed + trial if random_seed is not None else None
        # Must use the current isa_cap value for the Year init
        current_isa_cap = isa_cap # Assign the determined value
        year = Year(
            initial_inflation_rate=initial_inflation_rate,
            initial_unemployment_rate=initial_unemployment_rate,
            initial_isa_cap=current_isa_cap,
            initial_isa_threshold=isa_threshold,
            num_years=num_years
        )
        # Set seed for year progression if provided
        # year.next_year(random_seed=trial_seed) # No, seed is used *within* next_year

        # Create students with initial degrees and promotion potential
        students = _create_students(
             num_students, 
             initial_degrees_list, # Pass list of possible initial Degree objects
             initial_probs,        # Pass corresponding probabilities
             num_years,
             base_degrees_all,     # Pass dict of all Degree objects for lookup
             program_type,         # Pass program type for promotion logic
             # Pass params needed for promotion chance check within _create_students
             ecu_placement_prob, ecu_na_completion_prob, # Ecuador promotion checks
             guat_advancement_prob, # Guatemala promotion checks (guat_placement_prob removed)
             apply_graduation_delay
        )
        
        # Run the simulation and store results
        sim_results = simulate_simple(
            students=students,
            year=year, # Pass the initialized Year object
            num_years=num_years,
            limit_years=limit_years,
            isa_percentage=isa_percentage,
            performance_fee_pct=performance_fee_pct,
            gamma=False, # Consistent with assumptions in promotion/graduation
            price_per_student=price_per_student,
            new_malengo_fee=new_malengo_fee,
            annual_fee_per_student=annual_fee_per_student,
            apply_graduation_delay=apply_graduation_delay
        )
        df_list.append(sim_results)
        
        # Calculate and store statistics for this simulation
        stats = _calculate_simulation_statistics(
            students, num_students, num_years, limit_years
        )
        employment_stats.append(stats['employment_rate'])
        ever_employed_stats.append(stats['ever_employed_rate'])
        repayment_stats.append(stats['repayment_rate'])
        cap_stats.append(stats['cap_stats'])
        
        # Extract and store real (inflation-adjusted) payments
        # Ensure student.real_payments is used correctly
        real_payments_per_student = [student.real_payments for student in students]
        if real_payments_per_student: # Check if list is not empty
             total_payment[trial] = np.sum(pd.DataFrame(real_payments_per_student), axis=0)
        else:
             total_payment[trial] = np.zeros(num_years) # Handle case with 0 students

        investor_payment[trial] = sim_results['Investor_Real_Payments'] # Use results dict
        malengo_payment[trial] = sim_results['Malengo_Real_Payments'] # Use results dict
        
        # Extract and store nominal payments (not adjusted for inflation)
        nominal_payments_per_student = [student.payments for student in students]
        if nominal_payments_per_student:
             nominal_total_payment[trial] = np.sum(pd.DataFrame(nominal_payments_per_student), axis=0)
        else:
             nominal_total_payment[trial] = np.zeros(num_years)
             
        nominal_investor_payment[trial] = sim_results['Investor_Payments'] # Use results dict
        nominal_malengo_payment[trial] = sim_results['Malengo_Payments'] # Use results dict
        
        # Extract and store active students count
        active_students[trial] = sim_results['Active_Students_Count'] # Use results dict
    
    # Calculate summary statistics
    summary_stats = _calculate_summary_statistics(
        total_payment, investor_payment, malengo_payment,
        nominal_total_payment, nominal_investor_payment, nominal_malengo_payment,
        active_students,
        total_investment, 
        initial_degrees_list, # Pass initial degrees list
        initial_probs,        # Pass initial probabilities
        num_students,
        employment_stats, ever_employed_stats, repayment_stats, cap_stats,
        annual_fee_per_student
    )
    
    # Add simulation parameters to results
    summary_stats.update({
        'program_type': program_type,
        'total_investment': total_investment,
        'price_per_student': price_per_student,
        'isa_percentage': isa_percentage,
        'isa_threshold': isa_threshold,
        'isa_cap': isa_cap, # Use the final determined value
        'performance_fee_pct': performance_fee_pct,
        'annual_fee_per_student': annual_fee_per_student,
        # Add relevant program params used
        'ecu_year1_completion_prob': ecu_year1_completion_prob if program_type == 'Ecuador' else None,
        'ecu_placement_prob': ecu_placement_prob if program_type == 'Ecuador' else None,
        'ecu_na_completion_prob': ecu_na_completion_prob if program_type == 'Ecuador' else None,
        'guat_placement_prob': guat_placement_prob if program_type == 'Guatemala' else None,
        'guat_advancement_prob': guat_advancement_prob if program_type == 'Guatemala' else None,
    })
    
    return summary_stats

def _calculate_summary_statistics(
    total_payment: Dict[int, np.ndarray],
    investor_payment: Dict[int, np.ndarray],
    malengo_payment: Dict[int, np.ndarray],
    nominal_total_payment: Dict[int, np.ndarray],
    nominal_investor_payment: Dict[int, np.ndarray],
    nominal_malengo_payment: Dict[int, np.ndarray],
    active_students: Dict[int, np.ndarray],
    total_investment: float,
    initial_degrees: List[Degree], # Renamed parameter
    initial_probs: List[float],    # Renamed parameter
    num_students: int,
    employment_stats: List[float],
    ever_employed_stats: List[float],
    repayment_stats: List[float],
    cap_stats: List[Dict[str, Any]],
    annual_fee_per_student: float = 300
) -> Dict[str, Any]:
    """Helper function to calculate summary statistics across all simulations."""
    # Calculate summary statistics for real (inflation-adjusted) payments
    payments_df = pd.DataFrame(total_payment)
    # Ensure payments_df has columns even if num_sims=0 or 1
    if payments_df.empty and len(total_payment)>0: # Handle case where dict has keys but arrays are empty
        payments_df = pd.DataFrame(index=range(len(next(iter(total_payment.values()))))) # Use length of first array
    
    average_total_payment_per_sim = np.sum(payments_df, axis=0)
    average_total_payment = average_total_payment_per_sim.mean() if not average_total_payment_per_sim.empty else 0

    # Calculate weighted average duration (avoiding division by zero)
    payment_sums = np.sum(payments_df, axis=0) # Sum payments over years for each sim
    average_duration = 0
    if not payment_sums.empty and np.any(payment_sums > 0):
        # Convert to numpy arrays to avoid pandas indexing issues
        payments_np = payments_df.to_numpy()
        payment_sums_np = payment_sums.to_numpy()
        
        # Ensure weights calculation handles potential shape mismatches or empty arrays
        if payments_np.size > 0 and payment_sums_np.size > 0:
             weights = np.zeros_like(payments_np)
             # Iterate safely
             valid_sums_indices = np.where(payment_sums_np > 0)[0]
             if valid_sums_indices.size > 0:
                  weights[:, valid_sums_indices] = payments_np[:, valid_sums_indices] / payment_sums_np[valid_sums_indices]
             
             # Calculate weighted average
             years = np.arange(1, payments_np.shape[0] + 1)
             # Ensure broadcasting works or use explicit looping/summation
             # weighted_durations = np.sum(years[:, np.newaxis] * weights, axis=0) # Original approach
             # Safer alternative:
             if years.size == weights.shape[0]: # Check dimension match
                 weighted_durations = np.einsum('i,ij->j', years, weights) # Efficient sum-product
                 average_duration = np.mean(weighted_durations[valid_sums_indices]) if valid_sums_indices.size > 0 else 0
             else:
                 print("Warning: Dimension mismatch in duration calculation.") # Debug print
                 average_duration = 0 # Fallback
        else:
             average_duration = 0 # Fallback if arrays are empty
    
    # Calculate real IRR (safely handle negative values and zero investment)
    # Ensure total_investment is positive before division/log
    IRR = -0.1 # Default negative return
    if total_investment > 0 and average_total_payment > 0 and average_duration > 0:
        # Use max(1, average_total_payment) to avoid log(0) or log(<1) issues if payment is tiny
        IRR = np.log(max(1, average_total_payment) / total_investment) / average_duration
    elif total_investment == 0 and average_total_payment > 0:
         IRR = np.inf # Infinite IRR if cost is zero and return is positive
    # Else remains -0.1
    
    # Calculate real investor payments
    investor_payments_df = pd.DataFrame(investor_payment)
    if investor_payments_df.empty and len(investor_payment)>0:
         investor_payments_df = pd.DataFrame(index=range(len(next(iter(investor_payment.values())))))

    average_investor_payment_per_sim = np.sum(investor_payments_df, axis=0)
    average_investor_payment = average_investor_payment_per_sim.mean() if not average_investor_payment_per_sim.empty else 0
    
    # Calculate real Malengo payments
    malengo_payments_df = pd.DataFrame(malengo_payment)
    if malengo_payments_df.empty and len(malengo_payment)>0:
         malengo_payments_df = pd.DataFrame(index=range(len(next(iter(malengo_payment.values())))))
    average_malengo_payment_per_sim = np.sum(malengo_payments_df, axis=0)
    average_malengo_payment = average_malengo_payment_per_sim.mean() if not average_malengo_payment_per_sim.empty else 0
    
    # Calculate real investor IRR using total investment as base
    investor_IRR = -0.1
    if total_investment > 0 and average_investor_payment > 0 and average_duration > 0:
        investor_IRR = np.log(max(1, average_investor_payment) / total_investment) / average_duration
    elif total_investment == 0 and average_investor_payment > 0:
         investor_IRR = np.inf
    
    # Calculate active students statistics
    active_students_df = pd.DataFrame(active_students)
    if active_students_df.empty and len(active_students)>0:
         active_students_df = pd.DataFrame(index=range(len(next(iter(active_students.values())))))

    active_students_by_year = active_students_df.mean(axis=1) if not active_students_df.empty else pd.Series(dtype=float)
    max_active_students = active_students_by_year.max() if not active_students_by_year.empty else 0
    avg_active_students = active_students_by_year.mean() if not active_students_by_year.empty else 0
    active_students_pct = avg_active_students / max(1, num_students) # Avoid division by zero
    
    # Calculate annual Malengo revenue from active students (base fee only)
    annual_malengo_revenue = avg_active_students * annual_fee_per_student
    total_malengo_revenue = annual_malengo_revenue * len(active_students_by_year) if not active_students_by_year.empty else 0
    
    # Calculate real payment quantiles (IRR based on quantiles of total payment per sim)
    payment_quantiles = {}
    quantiles_to_calc = [0, 0.25, 0.5, 0.75, 1.0]
    total_payments_per_sim = np.sum(payments_df, axis=0) if not payments_df.empty else pd.Series(dtype=float)

    for quantile in quantiles_to_calc:
        quantile_payment = total_payments_per_sim.quantile(quantile) if not total_payments_per_sim.empty else 0
        q_irr = -0.1 - (0.1 * (1-quantile)) # Default negative, lower for lower quantiles
        if total_investment > 0 and quantile_payment > 0 and average_duration > 0:
            q_irr = np.log(max(1, quantile_payment) / total_investment) / average_duration
        elif total_investment == 0 and quantile_payment > 0:
             q_irr = np.inf
        payment_quantiles[quantile] = q_irr
            
    # Calculate real investor payment quantiles (IRR based on quantiles of investor payment per sim)
    investor_payment_quantiles = {}
    investor_payments_per_sim = np.sum(investor_payments_df, axis=0) if not investor_payments_df.empty else pd.Series(dtype=float)

    for quantile in quantiles_to_calc:
        investor_quantile_payment = investor_payments_per_sim.quantile(quantile) if not investor_payments_per_sim.empty else 0
        q_irr = -0.1 - (0.1 * (1-quantile)) 
        if total_investment > 0 and investor_quantile_payment > 0 and average_duration > 0:
            q_irr = np.log(max(1, investor_quantile_payment) / total_investment) / average_duration
        elif total_investment == 0 and investor_quantile_payment > 0:
             q_irr = np.inf
        investor_payment_quantiles[quantile] = q_irr
    
    # Prepare real payment data for plotting (average over simulations)
    payment_by_year = payments_df.mean(axis=1) if not payments_df.empty else pd.Series(dtype=float)
    investor_payment_by_year = investor_payments_df.mean(axis=1) if not investor_payments_df.empty else pd.Series(dtype=float)
    malengo_payment_by_year = malengo_payments_df.mean(axis=1) if not malengo_payments_df.empty else pd.Series(dtype=float)
    
    # Calculate average employment and repayment statistics
    avg_employment_rate = np.mean(employment_stats) if employment_stats else 0
    avg_ever_employed_rate = np.mean(ever_employed_stats) if ever_employed_stats else 0
    avg_repayment_rate = np.mean(repayment_stats) if repayment_stats else 0
    
    # Calculate average cap statistics
    avg_cap_stats = {
        'payment_cap_pct': np.mean([s['payment_cap_pct'] for s in cap_stats]) if cap_stats else 0,
        'years_cap_pct': np.mean([s['years_cap_pct'] for s in cap_stats]) if cap_stats else 0,
        'no_cap_paid_pct': np.mean([s['no_cap_paid_pct'] for s in cap_stats]) if cap_stats else 0,
        # Ensure avg_cap_value exists and handle potential NaN/None from individual sims
        'avg_cap_value': np.nanmean([s.get('avg_cap_value', np.nan) for s in cap_stats]) if cap_stats else 0
    }
    # Replace NaN with 0 if needed after nanmean
    avg_cap_stats['avg_cap_value'] = 0 if np.isnan(avg_cap_stats['avg_cap_value']) else avg_cap_stats['avg_cap_value']

    
    # Calculate initial degree counts and percentages
    initial_degree_counts = {}
    initial_degree_pcts = {}
    if num_students > 0 and initial_degrees and initial_probs and len(initial_degrees) == len(initial_probs):
        for i, degree in enumerate(initial_degrees): 
            count = initial_probs[i] * num_students 
            pct = initial_probs[i]
            if degree.name in initial_degree_counts:
                 initial_degree_counts[degree.name] += count
                 initial_degree_pcts[degree.name] += pct
            else:
                 initial_degree_counts[degree.name] = count
                 initial_degree_pcts[degree.name] = pct
    
    # Calculate summary statistics for nominal (non-inflation-adjusted) payments
    nominal_payments_df = pd.DataFrame(nominal_total_payment)
    if nominal_payments_df.empty and len(nominal_total_payment)>0:
         nominal_payments_df = pd.DataFrame(index=range(len(next(iter(nominal_total_payment.values())))))
    avg_nominal_total_payment_per_sim = np.sum(nominal_payments_df, axis=0) if not nominal_payments_df.empty else pd.Series(dtype=float)
    avg_nominal_total_payment = avg_nominal_total_payment_per_sim.mean() if not avg_nominal_total_payment_per_sim.empty else 0
    
    nominal_investor_payments_df = pd.DataFrame(nominal_investor_payment)
    if nominal_investor_payments_df.empty and len(nominal_investor_payment)>0:
          nominal_investor_payments_df = pd.DataFrame(index=range(len(next(iter(nominal_investor_payment.values())))))
    avg_nominal_investor_payment_per_sim = np.sum(nominal_investor_payments_df, axis=0) if not nominal_investor_payments_df.empty else pd.Series(dtype=float)
    avg_nominal_investor_payment = avg_nominal_investor_payment_per_sim.mean() if not avg_nominal_investor_payment_per_sim.empty else 0

    nominal_malengo_payments_df = pd.DataFrame(nominal_malengo_payment)
    if nominal_malengo_payments_df.empty and len(nominal_malengo_payment)>0:
         nominal_malengo_payments_df = pd.DataFrame(index=range(len(next(iter(nominal_malengo_payment.values())))))
    avg_nominal_malengo_payment_per_sim = np.sum(nominal_malengo_payments_df, axis=0) if not nominal_malengo_payments_df.empty else pd.Series(dtype=float)
    avg_nominal_malengo_payment = avg_nominal_malengo_payment_per_sim.mean() if not avg_nominal_malengo_payment_per_sim.empty else 0
    
    # Calculate nominal IRR values using the same average duration as real IRR
    nominal_IRR = -0.1
    if total_investment > 0 and avg_nominal_total_payment > 0 and average_duration > 0:
        nominal_IRR = np.log(max(1, avg_nominal_total_payment) / total_investment) / average_duration
    elif total_investment == 0 and avg_nominal_total_payment > 0:
         nominal_IRR = np.inf
        
    nominal_investor_IRR = -0.1
    if total_investment > 0 and avg_nominal_investor_payment > 0 and average_duration > 0:
        nominal_investor_IRR = np.log(max(1, avg_nominal_investor_payment) / total_investment) / average_duration
    elif total_investment == 0 and avg_nominal_investor_payment > 0:
         nominal_investor_IRR = np.inf
    
    # Calculate nominal payment quantiles
    nominal_payment_quantiles = {}
    nominal_total_payments_per_sim = np.sum(nominal_payments_df, axis=0) if not nominal_payments_df.empty else pd.Series(dtype=float)
    for quantile in quantiles_to_calc:
        nominal_quantile_payment = nominal_total_payments_per_sim.quantile(quantile) if not nominal_total_payments_per_sim.empty else 0
        q_irr = -0.1 - (0.1 * (1-quantile))
        if total_investment > 0 and nominal_quantile_payment > 0 and average_duration > 0:
            q_irr = np.log(max(1, nominal_quantile_payment) / total_investment) / average_duration
        elif total_investment == 0 and nominal_quantile_payment > 0:
             q_irr = np.inf
        nominal_payment_quantiles[quantile] = q_irr
            
    # Calculate nominal investor payment quantiles
    nominal_investor_payment_quantiles = {}
    nominal_investor_payments_per_sim = np.sum(nominal_investor_payments_df, axis=0) if not nominal_investor_payments_df.empty else pd.Series(dtype=float)
    for quantile in quantiles_to_calc:
        nominal_investor_quantile_payment = nominal_investor_payments_per_sim.quantile(quantile) if not nominal_investor_payments_per_sim.empty else 0
        q_irr = -0.1 - (0.1 * (1-quantile))
        if total_investment > 0 and nominal_investor_quantile_payment > 0 and average_duration > 0:
            q_irr = np.log(max(1, nominal_investor_quantile_payment) / total_investment) / average_duration
        elif total_investment == 0 and nominal_investor_quantile_payment > 0:
             q_irr = np.inf
        nominal_investor_payment_quantiles[quantile] = q_irr

    # Prepare nominal payment data for plotting
    nominal_payment_by_year = nominal_payments_df.mean(axis=1) if not nominal_payments_df.empty else pd.Series(dtype=float)
    nominal_investor_payment_by_year = nominal_investor_payments_df.mean(axis=1) if not nominal_investor_payments_df.empty else pd.Series(dtype=float)
    nominal_malengo_payment_by_year = nominal_malengo_payments_df.mean(axis=1) if not nominal_malengo_payments_df.empty else pd.Series(dtype=float)
    
    return {
        # Real (inflation-adjusted) IRR values
        'IRR': IRR,
        'investor_IRR': investor_IRR,
        'average_total_payment': average_total_payment,
        'average_investor_payment': average_investor_payment,
        'average_malengo_payment': average_malengo_payment,
        'average_duration': average_duration,
        'payment_by_year': payment_by_year,
        'investor_payment_by_year': investor_payment_by_year,
        'malengo_payment_by_year': malengo_payment_by_year,
        'payments_df': payments_df, # Optional: return full df for deeper analysis? Large object. Maybe omit.
        'investor_payments_df': investor_payments_df, # Omit?
        'malengo_payments_df': malengo_payments_df, # Omit?
        'payment_quantiles': payment_quantiles,
        'investor_payment_quantiles': investor_payment_quantiles,
        
        # Active students statistics
        'active_students_by_year': active_students_by_year,
        'max_active_students': max_active_students,
        'avg_active_students': avg_active_students,
        'active_students_pct': active_students_pct,
        'annual_malengo_revenue': annual_malengo_revenue,
        'total_malengo_revenue': total_malengo_revenue,
        
        # Nominal (non inflation-adjusted) IRR values
        'nominal_IRR': nominal_IRR,
        'nominal_investor_IRR': nominal_investor_IRR,
        'average_nominal_total_payment': avg_nominal_total_payment,
        'average_nominal_investor_payment': avg_nominal_investor_payment,
        'average_nominal_malengo_payment': avg_nominal_malengo_payment,
        'nominal_payment_by_year': nominal_payment_by_year,
        'nominal_investor_payment_by_year': nominal_investor_payment_by_year,
        'nominal_malengo_payment_by_year': nominal_malengo_payment_by_year,
        # 'nominal_payments_df': nominal_payments_df, # Omit?
        # 'nominal_investor_payments_df': nominal_investor_payments_df, # Omit?
        # 'nominal_malengo_payments_df': nominal_malengo_payments_df, # Omit?
        'nominal_payment_quantiles': nominal_payment_quantiles,
        'nominal_investor_payment_quantiles': nominal_investor_payment_quantiles,
        
        # Other statistics
        # Removed adjusted_mean_salary/stdev as they are ambiguous now
        'employment_rate': avg_employment_rate, # Note: For non-STAY_HOME students
        'ever_employed_rate': avg_ever_employed_rate, # Note: For non-STAY_HOME students
        'repayment_rate': avg_repayment_rate, # Note: For non-STAY_HOME students
        'cap_stats': avg_cap_stats,
        'degree_counts': initial_degree_counts, # Based on initial assignment
        'degree_pcts': initial_degree_pcts      # Based on initial assignment
    }


def _calculate_simulation_statistics(
    students: List[Student], 
    num_students: int, 
    num_years: int, 
    limit_years: int
) -> Dict[str, Any]:
    """
    Helper function to calculate statistics for a single simulation run.
    Refined to separate statistics for migrated vs STAY_HOME students where appropriate.
    """
    # Track statistics for this simulation
    non_stay_home_students = 0
    ever_employed_host_country = 0
    students_made_payments = 0 # Count of non-STAY_HOME students making payments
    
    students_hit_payment_cap = 0
    students_hit_years_cap = 0
    students_hit_no_cap_but_paid = 0 # Renamed for clarity

    total_post_grad_periods_non_stay_home = 0
    total_employment_periods_non_stay_home = 0
    
    # Track repayments by cap status (includes all students, though STAY_HOME pay 0)
    total_repayment_cap_hit = 0
    total_years_cap_hit = 0
    total_no_cap_hit = 0
    
    cap_values = [] # Track actual cap values when hit

    for student in students:
        is_stay_home = student.degree.name == 'STAY_HOME'
        
        if not is_stay_home:
            non_stay_home_students += 1
            
            student_ever_employed = False
            student_made_payment = False
            employment_periods = 0
            post_grad_periods = 0
            initial_grad_year = student.graduation_year 

            for i in range(num_years):
                # Check post-graduation status based on initial graduation year
                if i >= initial_grad_year:
                    post_grad_periods += 1
                    # Check employment status using the flag
                    if student.is_employed and i < len(student.earnings): # Safety check
                        student_ever_employed = True
                        employment_periods += 1
                
                # Check if student made payment in this year
                if i < len(student.payments) and student.payments[i] > 0:
                    student_made_payment = True
            
            total_post_grad_periods_non_stay_home += post_grad_periods
            total_employment_periods_non_stay_home += employment_periods

            if student_ever_employed:
                ever_employed_host_country += 1
            if student_made_payment:
                students_made_payments += 1

        # Cap status checks (apply to all students, STAY_HOME won't hit caps)
        student_total_real_payment = sum(student.real_payments) # Use real payments for analysis

        if student.hit_cap: # Payment cap hit
            students_hit_payment_cap += 1
            total_repayment_cap_hit += student_total_real_payment
            if student.cap_value_when_hit > 0: # Store only if value recorded
                cap_values.append(student.cap_value_when_hit)
        elif student.years_paid >= limit_years: # Years cap hit
            students_hit_years_cap += 1
            total_years_cap_hit += student_total_real_payment
        else: # No cap hit
             # Check if they made any payment at all (only relevant for non-STAY_HOME)
             if not is_stay_home and student_total_real_payment > 0:
                 students_hit_no_cap_but_paid += 1
                 total_no_cap_hit += student_total_real_payment
             # If is_stay_home or made no payments, they fall into "no cap hit" implicitly but don't contribute to totals/counts here.

    # Calculate averages (with safe division)
    avg_repayment_cap_hit = total_repayment_cap_hit / max(1, students_hit_payment_cap) if students_hit_payment_cap > 0 else 0
    avg_repayment_years_hit = total_years_cap_hit / max(1, students_hit_years_cap) if students_hit_years_cap > 0 else 0
    avg_repayment_no_cap = total_no_cap_hit / max(1, students_hit_no_cap_but_paid) if students_hit_no_cap_but_paid > 0 else 0
    
    # Calculate average annual employment rate for non-stay-home students
    avg_annual_employment_rate = total_employment_periods_non_stay_home / max(1, total_post_grad_periods_non_stay_home) if total_post_grad_periods_non_stay_home > 0 else 0
    
    # Calculate rates based on non-STAY_HOME students
    ever_employed_rate = ever_employed_host_country / max(1, non_stay_home_students) if non_stay_home_students > 0 else 0
    repayment_rate = students_made_payments / max(1, non_stay_home_students) if non_stay_home_students > 0 else 0

    # Calculate cap percentages based on *total* students
    payment_cap_pct = students_hit_payment_cap / max(1, num_students)
    years_cap_pct = students_hit_years_cap / max(1, num_students)
    # Percentage who hit no cap *and made payments* (out of total students)
    no_cap_paid_pct = students_hit_no_cap_but_paid / max(1, num_students) 
    # Percentage who never paid (includes STAY_HOME and non-payers) = 1 - payment_cap% - years_cap% - no_cap_paid%
    never_paid_pct = 1.0 - payment_cap_pct - years_cap_pct - no_cap_paid_pct

    # Calculate average cap value when hit
    avg_cap_value = sum(cap_values) / max(1, len(cap_values)) if cap_values else 0
    
    # Return statistics
    return {
        'employment_rate': avg_annual_employment_rate,  # Avg annual employment rate among non-STAY_HOME students
        'ever_employed_rate': ever_employed_rate,  # % of non-STAY_HOME students ever employed in host country
        'repayment_rate': repayment_rate, # % of non-STAY_HOME students who made any payments
        'cap_stats': {
            'payment_cap_count': students_hit_payment_cap,
            'years_cap_count': students_hit_years_cap,
            'no_cap_paid_count': students_hit_no_cap_but_paid, # Renamed
            'payment_cap_pct': payment_cap_pct, # Pct of *total* students
            'years_cap_pct': years_cap_pct, # Pct of *total* students
            'no_cap_paid_pct': no_cap_paid_pct, # Pct of *total* students
            'never_paid_pct': max(0, never_paid_pct), # Ensure non-negative
            'avg_repayment_cap_hit': avg_repayment_cap_hit, # Avg payment for those hitting payment cap
            'avg_repayment_years_hit': avg_repayment_years_hit, # Avg payment for those hitting years cap
            'avg_repayment_no_cap': avg_repayment_no_cap, # Avg payment for those who paid but hit no cap
            'avg_cap_value': avg_cap_value # Avg value of payment cap when hit
        }
    }


def _create_degree_definitions() -> Dict[str, Dict[str, Any]]:
    """
    Helper function to create degree definitions for Ecuador and Guatemala programs.
    Includes promotion paths and a STAY_HOME option.
    Years_to_complete includes assumed +1 year host country setup/language.
    """
    return {
        'NHA_ECU': {   # Ecuador -> Host Country: 1yr Nursing Home Assistant phase
            'name': 'NHA_ECU',
            'mean_earnings': 14_000,
            'stdev': 1_500,
            'experience_growth': 0.02,
            'years_to_complete': 2,     # 1 yr Ecu + 1 yr Host setup
            'promotes_to_degree_name': 'NA_ECU', # Potential promotion target
            'years_to_promotion': 2      # Promotes 2 years *after* completing NHA phase (i.e., at year 2+2=4 total)
        },
        'NA_ECU': {    # Ecuador -> Host Country: 2yr Nursing Assistant phase (Target of promotion)
            'name': 'NA_ECU',
            'mean_earnings': 18_000,   # Higher earnings in this phase
            'stdev': 1_500,
            'experience_growth': 0.03,
            'years_to_complete': 2,     # Duration of *this phase* if promoted into
             # No further promotion defined from NA_ECU
        },
        'HOSP_ENTRY': { # Guatemala -> Host Country: Entry Hospitality phase
            'name': 'HOSP_ENTRY',
            'mean_earnings': 13_000,
            'stdev': 1_200,
            'experience_growth': 0.04,
            'years_to_complete': 1,     # 0 yr Guat course + 1 yr Host setup
            'promotes_to_degree_name': 'HOSP_ADV',
            'years_to_promotion': 3      # Promotes 3 years *after* completing Entry phase (i.e., at year 1+3=4 total)
        },
        'HOSP_ADV': { # Guatemala -> Host Country: Advanced Hospitality phase (Target of promotion)
            'name': 'HOSP_ADV',
            'mean_earnings': 20_000,
            'stdev': 1_800,
            'experience_growth': 0.04,
            'years_to_complete': 3,     # Duration of *this phase* (e.g., takes 3 years in this role)
             # No further promotion defined
        },
        'STAY_HOME': { # Represents not migrating or dropping out early from home country
             'name': 'STAY_HOME',
             'mean_earnings': 3_600,    # Low local earnings (nominal)
             'stdev': 500,              # Small variation
             'experience_growth': 0.00, # No growth
             'years_to_complete': 0,     # No host country completion time
             # No promotion
        }
    }


def _create_students(
    num_students: int, 
    initial_degrees: List[Degree], # List of *possible initial* Degree objects
    initial_probs: List[float],    # Probabilities of those initial degrees
    num_years: int,
    base_degrees_all: Dict[str, Degree], # Pass dict of all Degree objects for lookup
    program_type: str, # Need program type for promotion probability logic
    # Ecuador params for promotion check
    ecu_placement_prob: float,
    ecu_na_completion_prob: float,
    # Guatemala params for promotion check
    # guat_placement_prob: float, # Not needed here, used in _setup
    guat_advancement_prob: float,
    apply_graduation_delay: bool = False
) -> List[Student]:
    """
    Helper function to create students, assign initial degrees, 
    and set up promotion details based on conditional probabilities.
    """
    students = []
    # Assign initial degrees based on the calculated probabilities
    # Ensure probabilities sum to 1 for np.random.choice
    if not np.isclose(sum(initial_probs), 1.0):
         print(f"Warning: Initial probabilities do not sum to 1: {sum(initial_probs)}. Normalizing.")
         initial_probs = np.array(initial_probs) / sum(initial_probs)

    if not initial_degrees: # Handle empty list case
         return []
         
    assigned_initial_degrees = np.random.choice(initial_degrees, size=num_students, p=initial_probs)
    
    for i in range(num_students):
        initial_degree = assigned_initial_degrees[i]
        student = Student(initial_degree, num_years)
        
        # Apply graduation delay if enabled (only for non-STAY_HOME)
        if apply_graduation_delay and student.degree.name != 'STAY_HOME':
            # Use the years_to_complete of the *initial* degree for delay calculation
            base_years = student.degree.years_to_complete
            student.graduation_year = _calculate_graduation_delay(base_years, student.degree.name)
        elif student.degree.name == 'STAY_HOME':
             student.graduation_year = 0 # Effectively graduates immediately in local context
        else: # No delay applied or STAY_HOME
            student.graduation_year = student.degree.years_to_complete 

        # Set up promotion if applicable to the initial degree
        if student.degree.promotes_to_degree_name and student.degree.years_to_promotion is not None:
            
            promotes = False # Default to no promotion
            # Check if this specific student actually gets the promotion based on program path probabilities
            if program_type == 'Ecuador' and student.degree.name == 'NHA_ECU':
                # Promotion NHA -> NA requires placement AND NA completion success.
                # These checks happen *after* the student has implicitly completed Year 1 Ecu (since they are NHA).
                # Simulate the conditional checks:
                if np.random.random() < ecu_placement_prob: # Was placed?
                    if np.random.random() < ecu_na_completion_prob: # Completed NA training?
                        promotes = True
                        
            elif program_type == 'Guatemala' and student.degree.name == 'HOSP_ENTRY':
                # Promotion ENTRY -> ADV requires advancement success.
                # This check happens *after* the student was placed (since they are HOSP_ENTRY).
                if np.random.random() < guat_advancement_prob: # Advanced?
                    promotes = True

            # If promotion check passed, set the student's promotion details
            if promotes:
                promo_degree_name = student.degree.promotes_to_degree_name
                if promo_degree_name in base_degrees_all:
                    student.promotion_degree = base_degrees_all[promo_degree_name]
                    # Promotion happens years_to_promotion *after* initial graduation year
                    student.promotion_year = student.graduation_year + student.degree.years_to_promotion
                else:
                    # Log error if the target promotion degree wasn't found in definitions
                    print(f"Error: Promotion degree '{promo_degree_name}' not found in definitions.") 

        students.append(student)
        
    return students


def main():
    """
    Command-line entrypoint for running simulations for Ecuador or Guatemala.
    """
    # Import only when needed
    import matplotlib.pyplot as plt
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ISA Monte Carlo simulations for Ecuador/Guatemala.')
    
    # Basic simulation parameters
    parser.add_argument('--program', type=str, required=True, choices=['Ecuador', 'Guatemala'],
                        help='Program type (Ecuador or Guatemala)')
    parser.add_argument('--students', type=int, default=200, help='Number of students to simulate')
    parser.add_argument('--sims', type=int, default=20, help='Number of simulations to run')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--graduation-delay', action='store_true', 
                        help='Apply realistic graduation delays based on degree type')
    parser.add_argument('--years', type=int, default=25, help='Number of years to simulate')
    
    # Ecuador program parameters
    parser.add_argument('--ecu-year1-completion', type=float, default=0.90,
                        help='Ecuador: Probability of completing year 1 (default: 0.90)')
    parser.add_argument('--ecu-placement', type=float, default=0.80,
                        help='Ecuador: Probability of placement in host country (default: 0.80)')
    parser.add_argument('--ecu-na-completion', type=float, default=0.85,
                        help='Ecuador: Probability of promotion NHA->NA if placed (default: 0.85)')
    
    # Guatemala program parameters
    parser.add_argument('--guat-placement', type=float, default=0.85,
                        help='Guatemala: Probability of placement in host country (default: 0.85)')
    parser.add_argument('--guat-advancement', type=float, default=0.40,
                        help='Guatemala: Probability of promotion ENTRY->ADV if placed (default: 0.40)')
        
    # Fee parameters
    parser.add_argument('--annual-fee', type=float, default=300,
                        help='Annual fee per active student (in USD, default: 300)')
    
    # ISA parameters
    parser.add_argument('--isa-pct', type=float, default=None, # Default 10% set in run_simple_simulation
                        help='ISA percentage (default: 0.10)')
    parser.add_argument('--isa-cap', type=float, default=None, # Defaults based on program price
                        help='ISA payment cap (defaults: Ecuador 27k, Guatemala 22.5k)')
    parser.add_argument('--isa-threshold', type=float, default=13_000,
                        help='ISA payment threshold (default: 13000)')
    
    args = parser.parse_args()
    
    # --- Run Simulation ---
    results = run_simple_simulation(
        program_type=args.program,
        num_students=args.students,
        num_sims=args.sims,
        num_years=args.years,  # Pass the years parameter
        initial_unemployment_rate=0.08,
        initial_inflation_rate=0.02,
        performance_fee_pct=0.025,
        ecu_year1_completion_prob=args.ecu_year1_completion,
        ecu_placement_prob=args.ecu_placement,
        ecu_na_completion_prob=args.ecu_na_completion,
        guat_placement_prob=args.guat_placement,
        guat_advancement_prob=args.guat_advancement,
        isa_percentage=args.isa_pct,
        isa_cap=args.isa_cap,
        isa_threshold=args.isa_threshold,
        annual_fee_per_student=args.annual_fee,
        random_seed=args.seed,
        apply_graduation_delay=args.graduation_delay
    )
    
    # --- Print Results ---
    print("\n--- Simulation Results ---")
    print(f"Program Type: {args.program}")
    print(f"Students per Simulation: {args.students}")
    print(f"Number of Simulations: {args.sims}")
    print(f"Apply Graduation Delay: {args.graduation_delay}")
    
    # Print program-specific pathway probabilities based on initial assignment
    print(f"\nInitial Pathway Assignment (% of Students):")
    if results['degree_counts']:
        for degree_name, pct in results['degree_pcts'].items():
            print(f"- {degree_name}: {pct*100:.1f}%")
    else:
        print("- No degrees assigned.")
        
    # Print derived probabilities for context
    print(f"\nDerived Probabilities (Based on Input Parameters):")
    if args.program == 'Ecuador':
        prob_stay_home = 1.0 - args.ecu_year1_completion
        prob_start_nha = args.ecu_year1_completion
        prob_placed_given_start_nha = args.ecu_placement 
        prob_promote_given_placed = args.ecu_na_completion 
        prob_promote_overall = prob_start_nha * prob_placed_given_start_nha * prob_promote_given_placed
        prob_placed_no_promote = prob_start_nha * prob_placed_given_start_nha * (1.0 - prob_promote_given_placed)
        prob_not_placed = prob_start_nha * (1.0 - prob_placed_given_start_nha)
        
        print(f"- Stay Home (Fail Yr1 Ecu): {prob_stay_home*100:.1f}%")
        print(f"- Start Host Phase (Complete Yr1 Ecu): {prob_start_nha*100:.1f}%")
        print(f"  - Placed & Promoted NHA->NA: {prob_promote_overall*100:.1f}%")
        print(f"  - Placed & Not Promoted (Remain NHA path): {prob_placed_no_promote*100:.1f}%")
        print(f"  - Not Placed (Remain NHA path): {prob_not_placed*100:.1f}%")
        # Sanity check: stay_home + promote + placed_no_promote + not_placed should be ~100%
        print(f"  (Check sum: {(prob_stay_home + prob_promote_overall + prob_placed_no_promote + prob_not_placed)*100:.1f}%)")


    elif args.program == 'Guatemala':
        prob_stay_home = 1.0 - args.guat_placement
        prob_start_entry = args.guat_placement
        prob_promote_given_placed = args.guat_advancement 
        prob_promote_overall = prob_start_entry * prob_promote_given_placed
        prob_entry_no_promote = prob_start_entry * (1.0 - prob_promote_given_placed)

        print(f"- Stay Home (Not Placed): {prob_stay_home*100:.1f}%")
        print(f"- Start Host Phase (Placed): {prob_start_entry*100:.1f}%")
        print(f"  - Promoted ENTRY->ADV: {prob_promote_overall*100:.1f}%")
        print(f"  - Not Promoted (Remain Entry path): {prob_entry_no_promote*100:.1f}%")
        # Sanity check
        print(f"  (Check sum: {(prob_stay_home + prob_promote_overall + prob_entry_no_promote)*100:.1f}%)")


    print(f"\nFinancial Metrics (Averages over {args.sims} sims):")
    print(f"- Price per Student: ${results['price_per_student']:.2f}")
    print(f"- Total Investment per Sim: ${results['total_investment']:.2f}")
    print(f"- Avg Total Repayment (Real): ${results['average_total_payment']:.2f}")
    print(f"- Avg Total Repayment (Nominal): ${results['average_nominal_total_payment']:.2f}")
    print(f"- Avg Investor Repayment (Real): ${results['average_investor_payment']:.2f}")
    print(f"- Avg Investor Repayment (Nominal): ${results['average_nominal_investor_payment']:.2f}")
    print(f"- Avg Malengo Revenue (Real): ${results['average_malengo_payment']:.2f}")
    print(f"- Avg Malengo Revenue (Nominal): ${results['average_nominal_malengo_payment']:.2f}")
    print(f"- Avg Payment Duration: {results['average_duration']:.2f} years")
    
    print(f"\nIRR Metrics (Averages over {args.sims} sims):")
    print(f"- Real Total IRR: {results['IRR']*100:.2f}%")
    print(f"- Nominal Total IRR: {results['nominal_IRR']*100:.2f}%")
    print(f"- Real Investor IRR: {results['investor_IRR']*100:.2f}%")
    print(f"- Nominal Investor IRR: {results['nominal_investor_IRR']*100:.2f}%")
    
    print("\nQuantiles of Investor IRR (Real):")
    if results['investor_payment_quantiles']:
         for q, irr_val in results['investor_payment_quantiles'].items():
             print(f"- {int(q*100)}th Percentile: {irr_val*100:.2f}%")
    else:
         print("- Not available.")

    print("\nStudent Outcome Metrics (Averages, Migrated Students Only):")
    print(f"- Annual Employment Rate: {results['employment_rate']*100:.2f}%")
    print(f"- Ever Employed in Host Country: {results.get('ever_employed_rate', 0)*100:.2f}%")
    print(f"- Made Any ISA Payment: {results['repayment_rate']*100:.2f}%")
        
    print("\nActive Students & Fees (Averages):")
    print(f"- Avg Active Students per Year: {results['avg_active_students']:.1f} ({results['active_students_pct']*100:.1f}% of total)")
    print(f"- Peak Avg Active Students: {results['max_active_students']:.1f}")
    print(f"- Avg Annual Malengo Revenue (Active Fee): ${results['annual_malengo_revenue']:.2f}")
    # print(f"- Total Malengo Fee Revenue: ${results['total_malengo_revenue']:.2f}") # This might be misleading

    print("\nCap Statistics (% of Total Students):")
    caps = results['cap_stats']
    print(f"- Hit Payment Cap: {caps.get('payment_cap_pct', 0)*100:.2f}% (Avg Payment: ${caps.get('avg_repayment_cap_hit', 0):.2f})")
    print(f"- Hit Years Cap: {caps.get('years_cap_pct', 0)*100:.2f}% (Avg Payment: ${caps.get('avg_repayment_years_hit', 0):.2f})")
    print(f"- Paid but No Cap: {caps.get('no_cap_paid_pct', 0)*100:.2f}% (Avg Payment: ${caps.get('avg_repayment_no_cap', 0):.2f})")
    print(f"- Never Paid (Incl. STAY_HOME): {caps.get('never_paid_pct', 0)*100:.2f}%")
    print(f"- Avg Cap Value When Hit: ${caps.get('avg_cap_value', 0):.2f}")
    
    # --- Create and Save Plots ---
    prog_name = args.program # Use for filenames/titles
    plot_years = range(1, args.years + 1) # Use simulation length from args

    try:
        # 1. Annual payments plot
        plt.figure(figsize=(10, 6))
        if not results['investor_payment_by_year'].empty:
             plt.plot(plot_years, results['investor_payment_by_year'], label='Investor (Real)')
             plt.plot(plot_years, results['malengo_payment_by_year'], label='Malengo (Real)')
             plt.plot(plot_years, results['nominal_investor_payment_by_year'], label='Investor (Nominal)', linestyle='--')
             plt.plot(plot_years, results['nominal_malengo_payment_by_year'], label='Malengo (Nominal)', linestyle='--')
        plt.xlabel('Year')
        plt.ylabel('Average Payment per Student Cohort')
        plt.title(f'{prog_name} Program - Annual Payments')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plot1_fname = f'{prog_name}_annual_payments.png'
        plt.savefig(plot1_fname)
        plt.close() # Close plot to free memory

        # 2. Cumulative returns plot
        plt.figure(figsize=(10, 6))
        if not results['investor_payment_by_year'].empty:
             real_cumulative = results['investor_payment_by_year'].cumsum()
             nominal_cumulative = results['nominal_investor_payment_by_year'].cumsum()
             plt.plot(plot_years, real_cumulative, label='Investor Returns (Real)')
             plt.plot(plot_years, nominal_cumulative, label='Investor Returns (Nominal)', linestyle='--')
        plt.axhline(y=results['total_investment'], color='black', linestyle='-', label='Initial Investment')
        
        # Calculate and plot breakeven points
        breakeven_year_real = None
        if not real_cumulative.empty:
            try:
                 breakeven_year_real = next(i + 1 for i, val in enumerate(real_cumulative) if val >= results['total_investment'])
                 plt.plot(breakeven_year_real, results['total_investment'], 'ro', markersize=8, label=f'Breakeven (Real): Year {breakeven_year_real}')
            except StopIteration: pass # No breakeven found

        breakeven_year_nominal = None
        if not nominal_cumulative.empty:
             try:
                 breakeven_year_nominal = next(i + 1 for i, val in enumerate(nominal_cumulative) if val >= results['total_investment'])
                 plt.plot(breakeven_year_nominal, results['total_investment'], 'go', markersize=6, label=f'Breakeven (Nominal): Year {breakeven_year_nominal}')
             except StopIteration: pass

        plt.xlabel('Year')
        plt.ylabel('Cumulative Returns per Student Cohort')
        plt.title(f'{prog_name} Program - Cumulative Investor Returns')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plot2_fname = f'{prog_name}_cumulative_returns.png'
        plt.savefig(plot2_fname)
        plt.close()

        # 3. Active students plot
        plt.figure(figsize=(10, 6))
        if not results['active_students_by_year'].empty:
            plt.plot(plot_years, results['active_students_by_year'], label='Active Students')
        plt.xlabel('Year')
        plt.ylabel('Average Number of Active Students')
        plt.title(f'{prog_name} Program - Active Students Over Time')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plot3_fname = f'{prog_name}_active_students.png'
        plt.savefig(plot3_fname)
        plt.close()
        
        print(f"\nPlots saved to {plot1_fname}, {plot2_fname}, and {plot3_fname}")

    except Exception as e:
        print(f"\nError generating plots: {e}")


if __name__ == "__main__":
    main() 