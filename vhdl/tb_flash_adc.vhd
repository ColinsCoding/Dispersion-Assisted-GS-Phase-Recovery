-- tb_flash_adc.vhd
-- Self-checking testbench: instantiates BOTH flash_adc_behavioral and
-- flash_adc_structural side by side, sweeps v_in across the full range,
-- and confirms they produce IDENTICAL digital codes -- behavioral and
-- structural are two different DESCRIPTIONS of the same SEMANTICS, and
-- this is the direct proof they actually agree, not just an assertion.

library ieee;
use ieee.std_logic_1164.all;

entity tb_flash_adc is
end entity tb_flash_adc;

architecture sim of tb_flash_adc is

    component flash_adc_behavioral is
        port (v_in : in real; v_ref : in real; code : out std_logic_vector(1 downto 0));
    end component;

    component flash_adc_structural is
        port (v_in : in real; v_ref : in real; code : out std_logic_vector(1 downto 0));
    end component;

    signal v_in_tb            : real := 0.0;
    constant v_ref_tb         : real := 5.0;
    signal code_behavioral    : std_logic_vector(1 downto 0);
    signal code_structural    : std_logic_vector(1 downto 0);

    signal n_checked, n_failed : integer := 0;

begin

    DUT_BEH: flash_adc_behavioral
        port map (v_in => v_in_tb, v_ref => v_ref_tb, code => code_behavioral);

    DUT_STRUCT: flash_adc_structural
        port map (v_in => v_in_tb, v_ref => v_ref_tb, code => code_structural);

    STIMULUS: process
        constant N_STEPS : integer := 201;   -- 0.00, 0.025, ..., 1.00 * v_ref -- crosses every threshold
    begin
        for i in 0 to N_STEPS - 1 loop
            v_in_tb <= v_ref_tb * real(i) / real(N_STEPS - 1);
            wait for 1 ns;

            n_checked <= n_checked + 1;
            if code_behavioral /= code_structural then
                n_failed <= n_failed + 1;
                report "FAIL v_in=" & real'image(v_in_tb)
                     severity error;
            end if;
        end loop;

        wait for 1 ns;
        if n_failed = 0 then
            report "PASS: all " & integer'image(n_checked)
                 & " v_in sweep points gave IDENTICAL behavioral/structural codes";
        else
            report "FAIL: " & integer'image(n_failed) & " / " & integer'image(n_checked) & " points mismatched";
        end if;
        wait;
    end process STIMULUS;

end architecture sim;
