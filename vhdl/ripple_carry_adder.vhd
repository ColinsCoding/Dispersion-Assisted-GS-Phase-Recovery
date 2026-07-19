-- ripple_carry_adder.vhd
-- Mirrors dgs/digital_logic.py:ripple_carry_add(a_bits, b_bits), which loops
--   for i in range(n): s, carry = full_adder(a[i], b[i], carry)
-- A Python `for` loop over fixed, known bounds is exactly what a VHDL
-- `generate` statement is for: it unrolls at *elaboration* time into N
-- full_adder instances wired carry-to-carry, the literal hardware the loop
-- describes (n adders in series, latency = n * one adder's gate delay --
-- see dgs/logic_timing.py:ripple_carry_delay for that timing model).

library ieee;
use ieee.std_logic_1164.all;

entity ripple_carry_adder is
    generic (
        N : integer := 8          -- bit width; LSB-first, matches the Python convention
    );
    port (
        a, b  : in  std_logic_vector(N-1 downto 0);
        cin   : in  std_logic;
        sum   : out std_logic_vector(N-1 downto 0);
        cout  : out std_logic
    );
end entity ripple_carry_adder;

architecture structural of ripple_carry_adder is

    component full_adder is
        port (
            a, b, cin : in  std_logic;
            sum       : out std_logic;
            cout      : out std_logic
        );
    end component;

    -- carry[0] = cin, carry[N] = final carry out -- N+1 nodes for N adders,
    -- the same "carry, out = 0, []" / "carry = ..." accumulator the Python
    -- loop threads through each iteration.
    signal carry : std_logic_vector(N downto 0);

begin
    carry(0) <= cin;
    cout     <= carry(N);

    -- the unrolled loop: one full_adder per bit position, exactly the Python
    -- "for i in range(n)" body, instantiated N times instead of executed N times.
    ADDER_CHAIN: for i in 0 to N-1 generate
        FA: full_adder
            port map (
                a    => a(i),
                b    => b(i),
                cin  => carry(i),
                sum  => sum(i),
                cout => carry(i+1)
            );
    end generate ADDER_CHAIN;

end architecture structural;
