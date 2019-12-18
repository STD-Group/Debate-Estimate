
for index = 0: 99
    [wave, Fs] = audioread(['..\dataset\train\positive\', ...
        num2str(index), '\audio.wav']);
    plot(wave);
    disp([num2str(index), ': ', num2str(Fs)]);

    [wave, Fs] = audioread(['..\dataset\train\negative\', ...
        num2str(index), '\audio.wav']);
    plot(wave);
    disp([num2str(index), ': ', num2str(Fs)]);
end