function IMAGES = whitener(IMAGES)

    N = size(IMAGES, 1);
    M = size(IMAGES, 3);

    [fx fy]=meshgrid(-N/2:N/2-1,-N/2:N/2-1);
    rho=sqrt(fx.*fx+fy.*fy);
    f_0=0.4*N;
    filt=rho.*exp(-(rho/f_0).^4);

    for i=1:M
        image=IMAGES(:,:,i);
        If=fft2(image);
        imagew=real(ifft2(If.*fftshift(filt)));
        IMAGES(:,:,i)=imagew;
    end

end