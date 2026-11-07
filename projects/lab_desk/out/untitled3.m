r = 2.0;       % meters
f = 0.2;       % cycles/ns
delay = r/c;

source = sin(2*pi*f*t);
received = sin(2*pi*f*(t-delay));

plot(t, source, 'b', 'LineWidth', 1.4); hold on
plot(t, received, 'r--', 'LineWidth', 1.4);
xlim([0 20])
grid on
legend('Source','Delayed received signal')
xlabel('Time (ns)')
ylabel('Amplitude')