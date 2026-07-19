-- full_adder.vhd
-- Mirrors dgs/digital_logic.py:full_adder(a, b, cin) -> (sum, cout), which is
-- itself built from two half adders -- the same composition done here at the
-- VHDL structural level via two `half_adder` component instances, not
-- re-derived from raw gates. The Python comment "full adder from two half
-- adders; cout = OR of the two carries" is the literal architecture below.

library ieee;
use ieee.std_logic_1164.all;

entity full_adder is
    port (
        a, b, cin : in  std_logic;
        sum       : out std_logic;
        cout      : out std_logic
    );
end entity full_adder;

architecture structural of full_adder is

    component half_adder is
        port (
            a     : in  std_logic;
            b     : in  std_logic;
            sum   : out std_logic;
            carry : out std_logic
        );
    end component;

    signal s1, c1, c2 : std_logic;

begin
    -- s1, c1 = half_adder(a, b)
    HA1: half_adder port map (a => a, b => b, sum => s1, carry => c1);

    -- sum, c2 = half_adder(s1, cin)
    HA2: half_adder port map (a => s1, b => cin, sum => sum, carry => c2);

    -- cout = c1 OR c2
    cout <= c1 or c2;

end architecture structural;
