-- flash_adc_structural.vhd
-- The SAME 2-bit flash ADC, STRUCTURAL style: describes HOW it is built --
-- three comparator.vhd instances (component instantiation = function
-- composition, same idea as full_adder.vhd built from two half_adders),
-- wired to a thermometer-to-binary encoder. This is closer to the
-- SYNTAX/topology of a real flash ADC (an actual chip has exactly this
-- comparator-ladder-plus-encoder architecture) than to an abstract
-- algorithm -- behavioral and structural describe the SAME semantics
-- (same input/output mapping) through two different lenses.

library ieee;
use ieee.std_logic_1164.all;

entity flash_adc_structural is
    port (
        v_in  : in  real;
        v_ref : in  real;
        code  : out std_logic_vector(1 downto 0)
    );
end entity flash_adc_structural;

architecture structural of flash_adc_structural is
    component comparator is
        port (
            v_in    : in  real;
            v_ref   : in  real;
            cmp_out : out std_logic
        );
    end component;

    signal c1, c2, c3 : std_logic;   -- thermometer code: c1<=c2<=c3 always, by construction
    signal therm       : std_logic_vector(2 downto 0);
    signal thresh1, thresh2, thresh3 : real;   -- named signals: this ModelSim version
                                                -- rejects inline expressions as `real`
                                                -- port-map actuals ("not globally static")
begin
    thresh1 <= 0.25 * v_ref;
    thresh2 <= 0.50 * v_ref;
    thresh3 <= 0.75 * v_ref;

    -- three comparators at 1/4, 1/2, 3/4 of v_ref -- the actual comparator
    -- ladder a real flash ADC chip is built from
    cmp1: comparator port map (v_in => v_in, v_ref => thresh1, cmp_out => c1);
    cmp2: comparator port map (v_in => v_in, v_ref => thresh2, cmp_out => c2);
    cmp3: comparator port map (v_in => v_in, v_ref => thresh3, cmp_out => c3);

    therm <= c3 & c2 & c1;

    -- thermometer-to-binary encoder: count how many comparators fired
    with therm select
        code <= "00" when "000",
                "01" when "001",
                "10" when "011",
                "11" when "111",
                "00" when others;   -- unreachable for a monotonic comparator ladder
end architecture structural;
